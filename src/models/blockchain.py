import numpy as np

class Epoch:
    def __init__(self):
        self.rewards = []  # To store the rewards of each block within an epoch
        self.hashrates = [] # To store the hashrates of each block within an epoch
        self.ceil = None
        self.floor = None
        self.difficulty = None
        self.elapsed_time = None # T_n

    def median_block_reward(self):
        sorted_rewards = sorted(self.rewards)
        return np.median(sorted_rewards) 

    def average_hashrate(self):
        sorted_hashrates = sorted(self.hashrates) 
        return np.mean(sorted_hashrates)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_hashrate(self, n):
        self.hashrates.append(n)


class Blockchain:
    def __init__(self, tau, gamma, upper_bound, lower_bound, initial_difficulty = 1e12):
        self.tau = tau
        self.gamma = gamma
        self.epochs = [Epoch()]
        self.epochs[-1].difficulty = initial_difficulty
        self.DT = 646325930.8894218  # Puzzle difficulty/blockchain growth rate
        self.DT_N_UB = upper_bound
        self.DT_N_LB = lower_bound

    def adjust_reward(self, reward, epoch_idx):
        epoch = self.epochs[epoch_idx]
        if epoch.ceil is not None:
            reward = min(reward, epoch.ceil)
        if epoch.floor is not None:
            reward = max(reward, epoch.floor)
        epoch.add_reward(reward)
        return reward
    
    def adjust_hashrate(self, hashrate, epoch_idx, e_current, P_current, efficiency_current, electricity_cost_current, new=True):
        alpha1 = 45.6767
        alpha2 = 0.4108
        alpha3 = -1.5100
        alpha4 = 0.2499
        alpha5 = 0.0346

        epoch = self.epochs[epoch_idx]
        log_eP_current = np.log(1+ (e_current * P_current)) 
        log_eff_current = np.log(1+efficiency_current)
        log_electricity_cost_current = np.log(1+electricity_cost_current)
        # Put it all together:
        # bN_{t+1} = α1 + α2*log(eP_t) + α3*log(efficiency_t) + α4*log(c_t)
        if new:
            predicted_log_hashrate = (
                alpha1 
                + alpha2 * log_eP_current
                + alpha3 * log_eff_current
            )
        else:
            predicted_log_hashrate = (
                alpha1 
                + alpha2 * log_eP_current
                + alpha3 * log_eff_current
                + alpha4 * log_electricity_cost_current
            )

        # Exponentiate to get N_{t+1}
        new_hashrate = np.exp(predicted_log_hashrate)      
        epoch.add_hashrate(new_hashrate)
        return new_hashrate
    
    def end_of_epoch(self):
        last_epoch = self.epochs[-1]
        P_B_median = last_epoch.median_block_reward()
        N_n = last_epoch.average_hashrate()
        new_epoch = Epoch()

        BLOCKS_PER_EPOCH = 14
        T_STAR = BLOCKS_PER_EPOCH * 10 * 60 # 1,209,600 for 2016 blocks
        D_n = last_epoch.difficulty
    
        T_n = (BLOCKS_PER_EPOCH * D_n * 2**32) / N_n
        last_epoch.elapsed_time = T_n

        T_hat = max(T_STAR /4, min(T_n, 4 *T_STAR))
        D_next = D_n * (T_STAR / T_hat)
        new_epoch.difficulty = D_next
        #print("\nNEW EPOCH\n")
        #print(f"Updating difficulty from {D_n} => {D_next}")
        #print("Median Block Reward: " + str(P_B_median) + '\n')

        # Case 1: Hashrate too high - need to decrease rewards
        if self.DT > self.DT_N_UB:
            #("Case 1: Hashrate above upper bound")
            new_epoch.ceil = (self.tau) * P_B_median
            new_epoch.floor = None  # Clear any existing floor
    
        # Case 2: Hashrate within bounds - gradual adjustment
        elif self.DT_N_LB < self.DT < self.DT_N_UB:
            
            # Previous epoch had a ceiling
            if last_epoch.ceil is not None:
                if last_epoch.ceil >= self.DT:
                    # Gradually relax the ceiling
                    new_epoch.ceil =  (1+self.gamma) * last_epoch.ceil
                    #("Old ceiling: " + str(last_epoch.ceil))
                    #("Ceiling relaxed: " + str(new_epoch.ceil))
                else:
                    # Remove ceiling if it's no longer needed
                    new_epoch.ceil = None
            
            # Previous epoch had a floor
            elif last_epoch.floor is not None:
                if last_epoch.floor <= self.DT:
                    # Gradually relax the floor
                    new_epoch.ceil = (self.gamma) * last_epoch.floor
                    #("Old floor: " + str(last_epoch.floor))
                    #("Floor relaxed: " + str(new_epoch.floor))
                else:
                    # Remove floor if it's no longer needed
                    new_epoch.floor = None
            else:
                new_epoch.floor = last_epoch.floor
                new_epoch.ceil = last_epoch.ceil

        # Case 3: Hashrate too low - need to increase rewards
        elif self.DT < self.DT_N_LB:
            #("Case 3: Hashrate below lower bound")
            new_epoch.floor =  (1+(1-self.tau)) * P_B_median
            new_epoch.ceil = None  # Clear any existing ceiling

        self.epochs.append(new_epoch)