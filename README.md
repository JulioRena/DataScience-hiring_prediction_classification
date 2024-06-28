# hiring_prediction_classification

### Este conjunto de dados fornece insights sobre os fatores que influenciam as decisões de contratação. Cada registro representa um candidato com diversos atributos considerados durante o processo de contratação. O objetivo é prever se um candidato será contratado com base nesses atributos.



### Age
Description: Age of the candidate.

Data Range: 20 to 50 years.
Data Type: Integer.

### Gender
Description: Gender of the candidate.
Categories: Male (0) or Female (1).
Data Type: Binary.

### Education Level
Description: Highest level of education attained by the candidate.
Categories:
1: Bachelor's (Type 1)
2: Bachelor's (Type 2)
3: Master's
4: PhD
Data Type: Categorical.

### Experience Years
Description: Number of years of professional experience.
Data Range: 0 to 15 years.
Data Type: Integer.

### Previous Companies Worked
Description: Number of previous companies where the candidate has worked.
Data Range: 1 to 5 companies.
Data Type: Integer.

### Distance From Company
Description: Distance in kilometers from the candidate's residence to the hiring company.
Data Range: 1 to 50 kilometers.
Data Type: Float (continuous).

### Interview Score
Description: Score achieved by the candidate in the interview process.
Data Range: 0 to 100.
Data Type: Integer.

### Skill Score
Description: Assessment score of the candidate's technical skills.
Data Range: 0 to 100.
Data Type: Integer.

### Personality Score
Description: Evaluation score of the candidate's personality traits.
Data Range: 0 to 100.
Data Type: Integer.

### Recruitment Strategy
Description: Strategy adopted by the hiring team for recruitment.
Categories:
1: Aggressive
2: Moderate
3: Conservative
Data Type: Categorical.

### Hiring Decision (Target Variable)
Description: Outcome of the hiring decision.
Categories:
0: Not hired
1: Hired
Data Type: Binary (Integer).
Dataset Information
Records: 1500
Features: 10
Target Variable: HiringDecision (Binary)
