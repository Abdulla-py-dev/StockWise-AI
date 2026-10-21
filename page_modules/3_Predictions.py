import streamlit as st
from components.sidebar import render_page_placeholder


def render() -> None:
    render_page_placeholder(
        "Predictions",
        "AI-powered expiry predictions will be implemented in a later phase using scikit-learn.",
    )
