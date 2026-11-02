# Report review notes

The main assignment report (`analysis_report.md`) is generated from the actual outputs by `src/04_generate_report.py`.

Before submitting, review these points and add your own interpretation where appropriate:

- Confirm the event window using `hourly_registration_summary.csv` and the generated time-series plots.
- Explain that the historical fraud labels are used only for retrospective event confirmation; the Isolation Forest demonstration does not use those labels.
- Explain the difference between precision and recall in the context of fraud investigation capacity and missed fraudulent registrations.
- Discuss the limitation of random splitting when a single time-localized incident exists.
- Do not claim that the model proves causation or that a flagged registration is confirmed fraud.
