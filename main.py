import streamlit as st
import base64
from llm_query import get_image_informations, farma_chatbot

if "language" not in st.session_state:
    st.session_state.language = "English"

if "sidebar_" not in st.session_state:
    st.session_state.sidebar_ = False

if "chatbot" not in st.session_state:
    st.session_state.chatbot = False

if "initial_upload" not in st.session_state:
    st.session_state.initial_upload = True

if "results_print" not in st.session_state:
    st.session_state.results_print = False

if "farma_chatbot" not in st.session_state:
    st.session_state.farma_chatbot = farma_chatbot()

if "results" not in st.session_state:
    st.session_state.results = {}


def sidebar_():
    with st.sidebar:
        st.title("Ask me your Agriculture doubts!!!!")


def main():
    content_placeholder = st.empty()
    chatbot_placeholder = st.empty()

    if st.session_state.initial_upload:
        with content_placeholder.container():
            st.title("FarmaCare")
            results = None
            file = st.file_uploader("Upload the image of the soil or an infected plant")

            if file is not None:
                try:
                    bytes_data = file.read()
                    base64_data = base64.b64encode(bytes_data).decode('utf-8')
                    results = get_image_informations(base64_data)

                    if not results:
                        st.write("Error: No valid data extracted from the image.")
                        st.stop()

                    # Check if the result is for soil or plant
                    if results.get("type") == "soil":
                        st.session_state.results = {
                            "Soil Type": results.get("soil_type"),
                            "Crops Suitable": results.get("crops_suitable"),
                            "Description": results.get("description"),
                            "type": "soil"
                        }
                    elif results.get("type") == "plant":
                        st.session_state.results = {
                            "Crop Type": results.get("crop_type"),
                            "Disease": results.get("disease"),
                            "Treatment/Prevention": results.get("treatment_prevention"),
                            "Description": results.get("description"),
                            "type": "plant"
                        }
                    else:
                        st.write("Unknown type detected:", results.get("type"))
                        st.stop()

                    st.session_state.results_print = True
                    st.session_state.initial_upload = False

                except Exception as e:
                    st.write(f"An error occurred: {e}")

    if st.session_state.results_print:
        content_placeholder.empty()
        with content_placeholder.container():
            if st.session_state.results.get("type") == "soil":
                st.header("_Soil Type_", divider="rainbow")
                st.write(st.session_state.results.get("Soil Type", "Not Available"))

                st.header("_Crops Suitable_", divider="rainbow")
                st.write(st.session_state.results.get("Crops Suitable", "Not Available"))

                st.header("_Description_", divider="rainbow")
                description = st.session_state.results.get("Description")
                if description and description.lower() != "none":
                    st.write(description)
                else:
                    st.write("No description available.")

            elif st.session_state.results.get("type") == "plant":
                st.header("_Crop Type_", divider="rainbow")
                st.write(st.session_state.results.get("Crop Type", "Not Available"))

                st.header("_Disease_", divider="rainbow")
                st.write(st.session_state.results.get("Disease", "Not Available"))

                st.header("_Treatment/Prevention_", divider="rainbow")
                st.write(st.session_state.results.get("Treatment/Prevention", "Not Available"))

                st.header("_Description_", divider="rainbow")
                st.write(st.session_state.results.get("Description", "Not Available"))

            if st.button("Let's Chat !!!!"):
                st.session_state.chatbot = True
                st.session_state.results_print = False
                st.session_state.sidebar_ = True

    if st.session_state.sidebar_:
        with st.sidebar:
            option = st.selectbox(
                "Select Language",
                ("English", "Hindi", "Kannada", "Tamil", "Malayalam", "Telugu")
            )
            language_dict = {
                "English": "en",
                "Hindi": "hi",
                "Kannada": "kn",
                "Tamil": "ta",
                "Malayalam": "ml",
                "Telugu": "te"
            }
            st.session_state.language = option

    if st.session_state.chatbot:
        content_placeholder.empty()
        st.title("FarmaCare Bot!!")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me something"):
            st.chat_message("user").markdown(prompt)
            st.session_state.farma_chatbot.update_user_message(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = st.session_state.farma_chatbot.chatbot_runner(st.session_state.results, prompt)
            st.session_state.farma_chatbot.update_ai_message(response)
            if st.session_state.language == "English":
                with st.chat_message("assistant"):
                    st.markdown(response)
            else:
                response_translated = st.session_state.farma_chatbot.translator_for_bot(response, st.session_state.language)
                with st.chat_message("assistant"):
                    st.markdown(response_translated)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
