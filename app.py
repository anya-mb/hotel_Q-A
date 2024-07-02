import streamlit as st
from streamlit_chat import message
from ask_travel_assistant import generate_response, set_titles_and_headers

set_titles_and_headers()

# Initialise session state variables
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# container for chat history
response_container = st.container()

# container for text box
container = st.container()




with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output = generate_response(
            user_input#, st.session_state["conversation_id"]
        )
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)


if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))


if st.button("Clear the chat"):
    st.session_state["generated"] = []
    st.session_state["past"] = []

    st.experimental_rerun()
