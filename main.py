import streamlit as st
from askPdf import app as askPdfApp
from askURL import app as askURLApp
from askYouTube import app as askYouTubeApp  # Import the new YouTube module

st.set_page_config(page_title="Chat with PDFs, Websites, and YouTube", layout="wide")

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.radio(
            'Choose an option',
            [app["title"] for app in self.apps]
        )

        for application in self.apps:
            if application["title"] == app:
                application["function"]()

if __name__ == "__main__":
    app = MultiApp()
    app.add_app("Chat with PDFs", askPdfApp)
    app.add_app("Chat with Websites", askURLApp)
    app.add_app("Chat with YouTube Transcripts", askYouTubeApp)  # Add the new option
    app.run()
