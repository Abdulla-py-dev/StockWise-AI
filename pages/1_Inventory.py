import streamlit as st
from components.sidebar import render_page_placeholder


def render() -> None:
    render_page_placeholder(
        "Inventory",
        "Inventory management (stock levels, batches, suppliers) will be implemented in a later phase.",
    )
