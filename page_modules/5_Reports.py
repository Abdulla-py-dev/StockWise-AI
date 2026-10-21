import streamlit as st
from components.sidebar import render_page_placeholder


def render() -> None:
    render_page_placeholder(
        "Reports",
        "PDF and CSV report generation will be implemented in a later phase.",
    )
