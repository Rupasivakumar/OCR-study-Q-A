import streamlit as st
from analyzer import analyze_nutrients, analyze_ingredients

st.set_page_config(page_title="NutriScan")
st.title("NutriScan")

st.subheader("Nutrition per 100 g")
col1, col2 = st.columns(2)
values = {
    "sugars": col1.number_input("Sugars (g)", min_value=0.0, value=0.0),
    "fat": col2.number_input("Total fat (g)", min_value=0.0, value=0.0),
    "saturated_fat": col1.number_input("Saturated fat (g)", min_value=0.0, value=0.0),
    "sodium": col2.number_input("Sodium (mg)", min_value=0.0, value=0.0),
}

ingredients = st.text_area("Ingredients", "Refined wheat flour (maida), sugar, palm oil, salt")

if st.button("Analyze"):
    result = analyze_nutrients(values)
    st.header(f"Grade: {result['grade']}")

    colors = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
    for r in result["ratings"].values():
        st.write(f"{colors[r['level']]} **{r['label']}**: {r['value']} {r['unit']} ({r['level']})")

    for note in result["notes"]:
        st.warning(note)

    st.subheader("Ingredient flags")
    flags = analyze_ingredients(ingredients)
    if not flags:
        st.write("Nothing notable found.")
    for f in flags:
        st.write(f"**{f['level']}**: {f['label']} {f['detail']}")