
<context>
- User's question: {{question}}
</context>


<task>
Detect the user's intent based on their question:
  -If the user's question is not related to bioinformatics, or is not related to omics data analysis, or is a general inquiry → call `"general_responder"`. 
  -Else, call `"env_checker"`.
</task>


<output_format>
Respond with a single valid JSON object in this format:
{
  "thought": "Summarize how the decision was made, including intent if inferred.",
  "output": "User-facing message to indicate your decision.",
  "next": "The next agent to call. MUST be determined strictly according to the decision_logic above. Choose from: env_checker or general_responder",
}
Do not include any other explanation or commentary outside the JSON object.
</output_format>