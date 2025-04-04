import streamlit as st

# This page exists only to create an "About" entry in the sidebar
# It immediately redirects to the main app page

st.set_page_config(page_title="About - Redirecting", page_icon="💻")

# Use JavaScript to redirect to the main page
st.markdown(
    """
    <script>
        window.location.href = '/';
    </script>
    <a href="/">Click here if you are not redirected automatically</a>
    """,
    unsafe_allow_html=True
)

# Also show a message in case the JavaScript redirect doesn't work
st.write("Redirecting to main page...") 