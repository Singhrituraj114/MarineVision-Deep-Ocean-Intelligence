from __future__ import annotations

from typing import Sequence

import streamlit as st

from utils.constants import CLASS_METADATA


def render_supported_classes(classes: Sequence[str], search_term: str = "") -> None:
    """Render searchable class explorer with category filtering and color coding."""
    
    st.markdown("<h2 class='section-title'>Supported Classes (34 Total Classes)</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-subtitle'>Complete label space used by the detector. 4 legacy metadata classes hidden by default.</p>",
        unsafe_allow_html=True,
    )

    # Exclude legacy classes by default
    legacy_classes = {"class_0", "class_1", "class_2", "class_3"}
    all_valid_classes = [label for label in classes if label not in legacy_classes]

    # Search + category filter controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_term = st.text_input(
            "Search classes",
            value=search_term,
            placeholder="🔍 Search classes (e.g. bottle, fish, net)…",
            label_visibility="collapsed",
            key="class_explorer_search",
        )

    normalized = search_term.strip().lower()
    filtered = [label for label in all_valid_classes if normalized in label.lower()] if normalized else list(all_valid_classes)

    with col2:
        selected_category = st.selectbox(
            "Filter by category",
            ["All Categories", "Marine Life", "Flora", "Equipment", "Containers", 
             "Textiles", "Metal", "Organic", "Paper", "String/Rope", "Plastic", "Unknown"],
            label_visibility="collapsed"
        )
    
    with col3:
        show_legacy = st.checkbox("Show Legacy Classes", value=False, help="Display corrupted Roboflow metadata classes")

    # Apply category filter
    if selected_category != "All Categories":
        filtered = [
            label for label in filtered
            if label in CLASS_METADATA and CLASS_METADATA[label].get("category") == selected_category
        ]

    if not filtered:
        st.info("No classes match your search. Try different keywords or category.")
        return

    # Display class count
    count_text = f"{len(filtered)} classes" if len(filtered) == len(all_valid_classes) else f"{len(filtered)} of {len(all_valid_classes)} classes"
    st.markdown(
        f"<div style='color: #9fc7d7; margin-bottom: 1rem'><strong>{count_text} found</strong></div>",
        unsafe_allow_html=True,
    )

    # Render filtered classes with correct indices (accounting for 4 legacy classes at start)
    html_chips = ""
    for label in filtered:
        # Find the actual index in the full class list
        actual_index = classes.index(label) + 1  # +1 for 1-based indexing
        metadata = CLASS_METADATA.get(label, {})
        emoji = metadata.get("emoji", "🏷️")
        category = metadata.get("category", "Other")
        
        # Single-line HTML: blank lines / indentation inside st.markdown are
        # parsed as Markdown code blocks and render as raw text.
        html_chips += (
            f'<div class="class-chip" title="{label}">'
            f'<div class="class-chip__index">{actual_index:02d}</div>'
            f'<div style="flex-grow: 1">'
            f'<div class="class-chip__label">{emoji} {label}</div>'
            f'<div style="font-size: 0.8rem; color: #9fc7d7; margin-top: 0.2rem">{category}</div>'
            f"</div></div>"
        )

    st.markdown(f"<div class='class-grid'>{html_chips}</div>", unsafe_allow_html=True)

    # Show legacy classes if enabled
    if show_legacy:
        st.divider()
        st.markdown(
            "<p style='color: #FF9500; font-size: 0.9rem'><strong>⚠️ Legacy Metadata Classes</strong><br>"
            "These four labels are corrupted Roboflow metadata classes retained only for model compatibility. "
            "They are not actual detection targets and should be ignored in predictions.</p>",
            unsafe_allow_html=True,
        )
        
        legacy_html = ""
        for idx, label in enumerate(["class_0", "class_1", "class_2", "class_3"], 1):
            metadata = CLASS_METADATA.get(label, {})
            emoji = metadata.get("emoji", "⚠️")
            category = metadata.get("category", "Legacy Metadata")
            
            legacy_html += (
                f'<div class="class-chip" title="{label}" style="opacity: 0.6">'
                f'<div class="class-chip__index">{idx:02d}</div>'
                f'<div style="flex-grow: 1">'
                f'<div class="class-chip__label">{emoji} {label}</div>'
                f'<div style="font-size: 0.8rem; color: #9fc7d7; margin-top: 0.2rem">{category}</div>'
                f"</div></div>"
            )
        
        st.markdown(f"<div class='class-grid'>{legacy_html}</div>", unsafe_allow_html=True)
