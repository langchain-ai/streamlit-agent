import streamlit as st


def st_feedback(ls_client, run_id, run_url, *, container=None, show_url=False):
    button_css = """
    .stChatMessage .stButton>button {
        border: 0px;
        border-radius: 1rem;
        padding: 0.1rem 0.5rem;
    }

    .stChatMessage .stButton>button:hover, .stChatMessage .stButton>button.dark:hover {
        border: 1px;
        border-color: var(--primary-color);
        cursor: pointer;
    }

    .stChatMessage .stButton>button.selected, .stChatMessage .stButton>button.dark.selected {
        background: var(--primary-color);
        border-color: var(--primary-color);
    }
    """
    st.markdown(f"<style>{button_css}</style>", unsafe_allow_html=True)

    container = container or st.container()
    with container:
        _, up, down, url = st.columns([9, 1, 1, 1])
        if up.button("👍", key=f"{run_id}_up"):
            ls_client.create_feedback(run_id, "user_score", score=1)
        if down.button("👎", key=f"{run_id}_down"):
            ls_client.create_feedback(run_id, "user_score", score=0)
        if show_url:
            url.markdown(f"[View run]({run_url})")


# TODO:
#   - Try clickable image of like / dislike https://github.com/vivien000/st-clickable-images
#   - Play with enabled / disabled or primary type for toggling https://docs.streamlit.io/library/advanced-features/button-behavior-and-examples
