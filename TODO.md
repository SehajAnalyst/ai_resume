# TODO - Final Project Strength Migration

- [ ] Search the entire project for "project_count".
- [ ] Remove every remaining occurrence of:
      - project_count
      - detect_projects()
      - result["project_count"]

- [ ] Ensure app.py displays:
      st.metric(
          "📁 Project Strength",
          f"{result['project_strength']}%"
      )

- [ ] Delete old ML artifacts:
      - rf_model.pkl
      - label_encoder.pkl

- [ ] Restart Streamlit:
      streamlit run app.py

- [ ] Verify new files are automatically recreated:
      - rf_model.pkl
      - label_encoder.pkl

- [ ] Test with 3 resumes:
      ✓ Resume with projects matching JD
      ✓ Resume with unrelated projects
      ✓ Resume without any projects

- [ ] Verify:
      - No errors occur.
      - Project Strength returns a percentage.
      - Final score changes according to Project Strength.
      - Suggestions use project_strength instead of project_count.

- [ ] Search entire project for:
      "Projects Found"

- [ ] Replace all UI labels with:
      "📁 Project Strength (%)"

- [ ] Commit changes:
      git add .
      git commit -m "Replace Projects Found with Project Strength metric"