# Data dictionary

_Generated from `core/schema.py` by `python -m core.schema` - do not edit by hand._

Questionnaire versions: graduate **v2**, employer **v2** (see `docs/questionnaires/questionnaire_revisions.md`).

**Role** says how each raw field is used: *Model feature* fields feed the predictors (directly or via a derived feature), *Outcome construction* fields build `duration_months`/`event` and are never predictors, *Fairness only* fields are used solely in the subgroup check, and *Descriptive* fields are reported but not modelled.

## 1. Graduate questionnaire - raw fields

| Field | Item | Type | Valid values | Role | Applies to | Nullable | Description |
|---|---|---|---|---|---|---|---|
| `respondent_id` | auto | id | free text | Identifier | all | no | Row identifier assigned at export |
| `survey_date` | auto | timestamp | ISO date-time | Outcome construction | all | no | Form submission timestamp; end of follow-up for graduates still searching |
| `age_group` | Q1 | categorical | `20-22`, `23-25`, `26-28`, `28+` | Descriptive only | all | yes | Age group |
| `gender` | Q2 | categorical | `male`, `female`, `prefer_not_to_say` | Fairness only | all | yes | Gender - fairness check ONLY, never a model feature |
| `university_name` | Q3 | text | free text | Descriptive only | all | no | Institution name; mapped to university_type via data/reference/university_types.csv |
| `degree_type` | Q4 | categorical | `computer_science`, `information_technology`, `software_engineering`, `information_systems`, `cyber_security`, `data_science`, `ai_ml`, `other` | Model feature (input) | all | no | Degree programme |
| `respondent_status` | Q5 | categorical | `graduated`, `final_year` | Outcome construction | all | no | Graduated vs final-year (derived from Q5) |
| `graduation_date` | Q5a | month | YYYY-MM | Outcome construction | graduated | no | Month of degree completion (final results released) - TIME ORIGIN |
| `expected_graduation_date` | Q5b | month | YYYY-MM | Descriptive only | final_year | no | Expected month of degree completion |
| `gpa` | Q6 | numeric | 0-4 | Model feature (input) | all | yes | GPA on a 4.0 scale |
| `degree_class` | Q6a | categorical | `first`, `second_upper`, `second_lower`, `third_or_pass` | Model feature (input) | all | yes | Degree class if no GPA; converted via DEGREE_CLASS_TO_GPA |
| `industry_modules` | Q7 | categorical | `yes`, `no`, `not_sure` | Descriptive only | all | yes | Degree included industry-oriented modules |
| `certifications_count` | Q9 | integer | 0-20 | Model feature (input) | all | no | Number of professional IT certifications |
| `internship_completed` | Q11 | binary | `yes`, `no` | Descriptive only | all | no | Completed an IT internship (yes/no) |
| `internship_count` | Q11a | integer | 0-5 | Descriptive only | all | no | Number of internships (0 if Q11 = No) |
| `internship_months` | Q12 | integer | 0-36 | Model feature (input) | all | no | Total months of IT internship experience |
| `project_count` | Q13 | integer | 0-30 | Model feature (input) | all | no | Personal/academic IT projects outside coursework |
| `has_portfolio` | Q14 | binary | `yes`, `no` | Candidate predictor (extended set only) | all | yes | Had a technical portfolio (GitHub, website) before completing the degree |
| `prep_start` | Q20 | categorical | `before_university`, `first_year`, `second_third_year`, `final_year`, `after_graduation`, `not_started` | Candidate predictor (extended set only) | all | yes | When they started preparing for IT employment |
| `self_learning_freq` | Q21 | likert | 1-5 | Candidate predictor (extended set only) | all | yes | Self-learning outside coursework during the degree (1 = never, 5 = very frequently) |
| `hackathon_count` | Q23 | integer | 0-30 | Candidate predictor (extended set only) | all | yes | Hackathons / coding competitions / technical events before completing the degree |
| `open_source` | Q24 | binary | `yes`, `no` | Candidate predictor (extended set only) | all | yes | Contributed to open-source projects before completing the degree |
| `sdlc_familiarity` | Q29 | likert | 1-5 | Candidate predictor (extended set only) | all | yes | Familiarity with SDLC practices at degree completion (1-5) |
| `employment_status` | Q16 | categorical | `full_time_it`, `it_internship`, `seeking_it`, `non_it`, `further_studies`, `self_employed`, `not_seeking` | Outcome construction | all | yes | Current status; used only to identify graduates not in the IT job market |
| `alt_path_start_date` | Q16a | month | YYYY-MM | Outcome construction | graduated | yes | If no IT role yet: start month of the current non-IT job, self-employment or further studies (optional; competing-risks analysis only) |
| `ever_it_job` | Q17 | binary | `yes`, `no` | Outcome construction | graduated | yes | Has ever started a paid IT role (excl. degree internships) - EVENT INDICATOR |
| `first_job_start_date` | Q17a | month | YYYY-MM | Outcome construction | graduated | yes | Start month of first paid IT role |
| `tech_programming_fundamentals` | Q18.1 | likert | 1-5 | Model feature (input) | all | yes | Programming Fundamentals - self-rating at degree completion (1-5) |
| `tech_oop` | Q18.2 | likert | 1-5 | Model feature (input) | all | yes | Object-Oriented Programming - self-rating at degree completion (1-5) |
| `tech_java` | Q18.3 | likert | 1-5 | Model feature (input) | all | yes | Java - self-rating at degree completion (1-5) |
| `tech_python` | Q18.4 | likert | 1-5 | Model feature (input) | all | yes | Python - self-rating at degree completion (1-5) |
| `tech_javascript` | Q18.5 | likert | 1-5 | Model feature (input) | all | yes | JavaScript - self-rating at degree completion (1-5) |
| `tech_sql_databases` | Q18.6 | likert | 1-5 | Model feature (input) | all | yes | SQL / Database Management - self-rating at degree completion (1-5) |
| `tech_git` | Q18.7 | likert | 1-5 | Model feature (input) | all | yes | Git / GitHub - self-rating at degree completion (1-5) |
| `tech_software_testing` | Q18.8 | likert | 1-5 | Model feature (input) | all | yes | Software Testing - self-rating at degree completion (1-5) |
| `tech_rest_apis` | Q18.9 | likert | 1-5 | Model feature (input) | all | yes | REST APIs - self-rating at degree completion (1-5) |
| `tech_web_development` | Q18.10 | likert | 1-5 | Model feature (input) | all | yes | Web Development - self-rating at degree completion (1-5) |
| `tech_cloud_computing` | Q18.11 | likert | 1-5 | Model feature (input) | all | yes | Cloud Computing - self-rating at degree completion (1-5) |
| `tech_linux` | Q18.12 | likert | 1-5 | Model feature (input) | all | yes | Linux / Command Line - self-rating at degree completion (1-5) |
| `tech_docker` | Q18.13 | likert | 1-5 | Model feature (input) | all | yes | Docker - self-rating at degree completion (1-5) |
| `tech_devops_cicd` | Q18.14 | likert | 1-5 | Model feature (input) | all | yes | DevOps / CI-CD - self-rating at degree completion (1-5) |
| `tech_cybersecurity` | Q18.15 | likert | 1-5 | Model feature (input) | all | yes | Cybersecurity Fundamentals - self-rating at degree completion (1-5) |
| `tech_data_analytics` | Q18.16 | likert | 1-5 | Model feature (input) | all | yes | Data Analytics - self-rating at degree completion (1-5) |
| `tech_ai_ml` | Q18.17 | likert | 1-5 | Model feature (input) | all | yes | Artificial Intelligence / Machine Learning - self-rating at degree completion (1-5) |
| `soft_communication` | Q19.1 | likert | 1-5 | Model feature (input) | all | yes | Communication Skills - self-rating at degree completion (1-5) |
| `soft_teamwork` | Q19.2 | likert | 1-5 | Model feature (input) | all | yes | Teamwork - self-rating at degree completion (1-5) |
| `soft_problem_solving` | Q19.3 | likert | 1-5 | Model feature (input) | all | yes | Problem Solving - self-rating at degree completion (1-5) |
| `soft_critical_thinking` | Q19.4 | likert | 1-5 | Model feature (input) | all | yes | Critical Thinking - self-rating at degree completion (1-5) |
| `soft_adaptability` | Q19.5 | likert | 1-5 | Model feature (input) | all | yes | Adaptability - self-rating at degree completion (1-5) |
| `soft_time_management` | Q19.6 | likert | 1-5 | Model feature (input) | all | yes | Time Management - self-rating at degree completion (1-5) |
| `soft_professionalism` | Q19.7 | likert | 1-5 | Model feature (input) | all | yes | Professionalism - self-rating at degree completion (1-5) |
| `soft_self_learning` | Q19.8 | likert | 1-5 | Model feature (input) | all | yes | Self-learning Ability - self-rating at degree completion (1-5) |
| `soft_leadership` | Q19.9 | likert | 1-5 | Model feature (input) | all | yes | Leadership Ability - self-rating at degree completion (1-5) |

## 2. Derived features

| Feature | Construction |
|---|---|
| `gpa` | Q6, or DEGREE_CLASS_TO_GPA[Q6a] when Q6 is blank |
| `degree_group` | DEGREE_GROUPS[degree_type] |
| `university_type` | lookup of university_name in data/reference/university_types.csv |
| `technical_mean` | mean of the 17 Q18 items |
| `soft_mean` | mean of the 9 Q19 items |
| `area_programming` | mean of programming_fundamentals, oop, java, python, javascript |
| `area_web_apis` | mean of web_development, rest_apis |
| `area_data_ai` | mean of sql_databases, data_analytics, ai_ml |
| `area_cloud_devops` | mean of cloud_computing, linux, docker, devops_cicd, git |
| `area_quality_security` | mean of software_testing, cybersecurity |
| `duration_months` | months from graduation_date to first_job_start_date (event) or survey_date (censored) |
| `event` | 1 if ever_it_job, else 0 |
| `has_portfolio / open_source` | yes -> 1, no -> 0 (extended feature set only) |
| `prep_years` | PREP_YEARS[prep_start]: years of preparation before completion (extended feature set only) |
| `gpa_3plus` | 1 if GPA >= 3.0 else 0 (transformed feature set only) |
| `log_internship_months / log_certifications / log_projects` | log(1 + count) (transformed feature set only) |

## 3. Model features

14 features -> **16 model columns** after one-hot encoding (reference levels: first listed). Skill blocks enter as **means**, not PCA.

| Feature | Type | Levels |
|---|---|---|
| `gpa` | numeric | - |
| `degree_group` | categorical | computing, it_is, specialised |
| `university_type` | categorical | state, non_state, foreign_affiliated |
| `internship_months` | numeric | - |
| `certifications_count` | numeric | - |
| `project_count` | numeric | - |
| `technical_mean` | numeric | - |
| `soft_mean` | numeric | - |
| `has_portfolio` | numeric | - |
| `prep_years` | numeric | - |
| `self_learning_freq` | numeric | - |
| `hackathon_count` | numeric | - |
| `open_source` | numeric | - |
| `sdlc_familiarity` | numeric | - |

Never model features: `gender` (fairness only) and every outcome / post-employment field: `current_role`, `duration_months`, `employer_name`, `employment_status`, `event`, `ever_it_job`, `first_job_start_date`, `job_title`, `salary`, `survey_date`.

## 4. Skill vocabulary and areas

| Key | Kind | Label | Graduate item | Employer item | Area |
|---|---|---|---|---|---|
| `programming_fundamentals` | technical | Programming Fundamentals | Q18.1 | Q8.1 | Programming |
| `oop` | technical | Object-Oriented Programming | Q18.2 | Q8.2 | Programming |
| `java` | technical | Java | Q18.3 | Q8.3 | Programming |
| `python` | technical | Python | Q18.4 | Q8.4 | Programming |
| `javascript` | technical | JavaScript | Q18.5 | Q8.5 | Programming |
| `sql_databases` | technical | SQL / Database Management | Q18.6 | Q8.6 | Data & AI |
| `git` | technical | Git / GitHub | Q18.7 | Q8.7 | Cloud & DevOps |
| `software_testing` | technical | Software Testing | Q18.8 | Q8.8 | Quality & Security |
| `rest_apis` | technical | REST APIs | Q18.9 | Q8.9 | Web & APIs |
| `web_development` | technical | Web Development | Q18.10 | Q8.10 | Web & APIs |
| `cloud_computing` | technical | Cloud Computing | Q18.11 | Q8.11 | Cloud & DevOps |
| `linux` | technical | Linux / Command Line | Q18.12 | Q8.12 | Cloud & DevOps |
| `docker` | technical | Docker | Q18.13 | Q8.13 | Cloud & DevOps |
| `devops_cicd` | technical | DevOps / CI-CD | Q18.14 | Q8.14 | Cloud & DevOps |
| `cybersecurity` | technical | Cybersecurity Fundamentals | Q18.15 | Q8.15 | Quality & Security |
| `data_analytics` | technical | Data Analytics | Q18.16 | Q8.16 | Data & AI |
| `ai_ml` | technical | Artificial Intelligence / Machine Learning | Q18.17 | Q8.17 | Data & AI |
| `communication` | soft | Communication Skills | Q19.1 | Q9.1 | Soft skills |
| `teamwork` | soft | Teamwork | Q19.2 | Q9.2 | Soft skills |
| `problem_solving` | soft | Problem Solving | Q19.3 | Q9.3 | Soft skills |
| `critical_thinking` | soft | Critical Thinking | Q19.4 | Q9.4 | Soft skills |
| `adaptability` | soft | Adaptability | Q19.5 | Q9.5 | Soft skills |
| `time_management` | soft | Time Management | Q19.6 | Q9.6 | Soft skills |
| `professionalism` | soft | Professionalism | Q19.7 | Q9.7 | Soft skills |
| `self_learning` | soft | Self-learning Ability | Q19.8 | Q9.8 | Soft skills |
| `leadership` | soft | Leadership Ability | Q19.9 | Q9.9 | Soft skills |

## 5. Actionable features (what-if)

| Feature | Min | Max | Step | Applies to | Max change |
|---|---|---|---|---|---|
| `internship_months` | 0 | 24 | 1 | all | - |
| `certifications_count` | 0 | 10 | 1 | all | - |
| `project_count` | 0 | 15 | 1 | all | - |
| `gpa` | 0 | 4 | 0.05 | final_year | 0.3 |
| `area_programming` | 1 | 5 | 0.5 | all | - |
| `area_web_apis` | 1 | 5 | 0.5 | all | - |
| `area_data_ai` | 1 | 5 | 0.5 | all | - |
| `area_cloud_devops` | 1 | 5 | 0.5 | all | - |
| `area_quality_security` | 1 | 5 | 0.5 | all | - |
| `area_soft` | 1 | 5 | 0.5 | all | - |

Not actionable: gpa (graduates), degree_group, university_type, gender. Changing a skill-area rating sets every item in that area to the new value before the block means are recomputed.

## 6. Employer questionnaire - raw fields

| Field | Item | Type | Valid values | Role | Applies to | Nullable | Description |
|---|---|---|---|---|---|---|---|
| `employer_id` | auto | id | free text | Identifier | all | no | Row identifier |
| `org_type` | Q1 | categorical | `software_dev`, `it_consulting`, `fintech`, `banking`, `telecom`, `bpo_kpo`, `government_it`, `startup`, `other` | Descriptive only | all | no | Organisation type |
| `org_size` | Q2 | categorical | `<50`, `50-200`, `201-500`, `500+` | Descriptive only | all | no | Organisation size |
| `factor_degree_qualification` | Q7.1 | likert | 1-5 | Descriptive only | all | yes | Importance of degree_qualification (1-5) |
| `factor_gpa` | Q7.2 | likert | 1-5 | Descriptive only | all | yes | Importance of gpa (1-5) |
| `factor_technical_skills` | Q7.3 | likert | 1-5 | Descriptive only | all | yes | Importance of technical_skills (1-5) |
| `factor_internship_experience` | Q7.4 | likert | 1-5 | Descriptive only | all | yes | Importance of internship_experience (1-5) |
| `factor_personal_projects` | Q7.5 | likert | 1-5 | Descriptive only | all | yes | Importance of personal_projects (1-5) |
| `factor_certifications` | Q7.6 | likert | 1-5 | Descriptive only | all | yes | Importance of certifications (1-5) |
| `factor_portfolio` | Q7.7 | likert | 1-5 | Descriptive only | all | yes | Importance of portfolio (1-5) |
| `factor_communication` | Q7.8 | likert | 1-5 | Descriptive only | all | yes | Importance of communication (1-5) |
| `factor_interview_performance` | Q7.9 | likert | 1-5 | Descriptive only | all | yes | Importance of interview_performance (1-5) |
| `imp_programming_fundamentals` | Q8.1 | likert | 1-5 | Descriptive only | all | yes | Importance of Programming Fundamentals (1-5) |
| `imp_oop` | Q8.2 | likert | 1-5 | Descriptive only | all | yes | Importance of Object-Oriented Programming (1-5) |
| `imp_java` | Q8.3 | likert | 1-5 | Descriptive only | all | yes | Importance of Java (1-5) |
| `imp_python` | Q8.4 | likert | 1-5 | Descriptive only | all | yes | Importance of Python (1-5) |
| `imp_javascript` | Q8.5 | likert | 1-5 | Descriptive only | all | yes | Importance of JavaScript (1-5) |
| `imp_sql_databases` | Q8.6 | likert | 1-5 | Descriptive only | all | yes | Importance of SQL / Database Management (1-5) |
| `imp_git` | Q8.7 | likert | 1-5 | Descriptive only | all | yes | Importance of Git / GitHub (1-5) |
| `imp_software_testing` | Q8.8 | likert | 1-5 | Descriptive only | all | yes | Importance of Software Testing (1-5) |
| `imp_rest_apis` | Q8.9 | likert | 1-5 | Descriptive only | all | yes | Importance of REST APIs (1-5) |
| `imp_web_development` | Q8.10 | likert | 1-5 | Descriptive only | all | yes | Importance of Web Development (1-5) |
| `imp_cloud_computing` | Q8.11 | likert | 1-5 | Descriptive only | all | yes | Importance of Cloud Computing (1-5) |
| `imp_linux` | Q8.12 | likert | 1-5 | Descriptive only | all | yes | Importance of Linux / Command Line (1-5) |
| `imp_docker` | Q8.13 | likert | 1-5 | Descriptive only | all | yes | Importance of Docker (1-5) |
| `imp_devops_cicd` | Q8.14 | likert | 1-5 | Descriptive only | all | yes | Importance of DevOps / CI-CD (1-5) |
| `imp_cybersecurity` | Q8.15 | likert | 1-5 | Descriptive only | all | yes | Importance of Cybersecurity Fundamentals (1-5) |
| `imp_data_analytics` | Q8.16 | likert | 1-5 | Descriptive only | all | yes | Importance of Data Analytics (1-5) |
| `imp_ai_ml` | Q8.17 | likert | 1-5 | Descriptive only | all | yes | Importance of Artificial Intelligence / Machine Learning (1-5) |
| `imp_communication` | Q9.1 | likert | 1-5 | Descriptive only | all | yes | Importance of Communication Skills (1-5) |
| `imp_teamwork` | Q9.2 | likert | 1-5 | Descriptive only | all | yes | Importance of Teamwork (1-5) |
| `imp_problem_solving` | Q9.3 | likert | 1-5 | Descriptive only | all | yes | Importance of Problem Solving (1-5) |
| `imp_critical_thinking` | Q9.4 | likert | 1-5 | Descriptive only | all | yes | Importance of Critical Thinking (1-5) |
| `imp_adaptability` | Q9.5 | likert | 1-5 | Descriptive only | all | yes | Importance of Adaptability (1-5) |
| `imp_time_management` | Q9.6 | likert | 1-5 | Descriptive only | all | yes | Importance of Time Management (1-5) |
| `imp_professionalism` | Q9.7 | likert | 1-5 | Descriptive only | all | yes | Importance of Professionalism (1-5) |
| `imp_self_learning` | Q9.8 | likert | 1-5 | Descriptive only | all | yes | Importance of Self-learning Ability (1-5) |
| `imp_leadership` | Q9.9 | likert | 1-5 | Descriptive only | all | yes | Importance of Leadership Ability (1-5) |

## 7. Questionnaire-study items (not model inputs)

Analysed in `core/survey_study.py` (report: `artifacts/questionnaire_study.md`). `multi` fields hold the selected options as semicolon-separated keys. Items that repeat a model question in other words are used as consistency checks.

### Graduate questionnaire

| Field | Item | Type | Valid values | Role | Applies to | Nullable | Description |
|---|---|---|---|---|---|---|---|
| `instruction_medium` | Q7a | categorical | `english`, `sinhala`, `tamil`, `mixed` | Questionnaire study only | all | yes | Main medium of instruction in the degree |
| `english_proficiency` | Q7b | likert | 1-5 | Questionnaire study only | all | yes | English proficiency (speaking and writing) at degree completion (1-5) |
| `degree_relevance` | Q8 | likert | 1-5 | Questionnaire study only | all | yes | Perceived relevance of the degree to industry (1-5) |
| `cert_areas` | Q10 | multi | any of `cloud`, `cybersecurity`, `networking`, `database`, `programming`, `data_ai`, `agile_scrum`, `other` | Questionnaire study only | all | yes | Certification areas completed |
| `portfolio_update_freq` | Q15 | categorical | `never`, `rarely`, `occasionally`, `frequently` | Questionnaire study only | all | yes | How often the technical portfolio is updated |
| `learning_platforms` | Q22 | multi | any of `coursera`, `udemy`, `linkedin_learning`, `google`, `microsoft_learn`, `aws_training`, `youtube`, `ai_tools`, `other` | Questionnaire study only | all | yes | Online learning platforms used |
| `languages` | Q25 | multi | any of `java`, `python`, `javascript`, `c_cpp`, `csharp`, `php`, `kotlin`, `other` | Questionnaire study only | all | yes | Programming languages used confidently (repeats Q18 Java/Python/JavaScript) |
| `project_areas` | Q26 | multi | any of `web`, `mobile`, `desktop`, `data_analytics`, `machine_learning`, `cloud`, `cybersecurity`, `iot`, `none` | Questionnaire study only | all | yes | Development areas with completed projects |
| `problem_confidence` | Q27 | likert | 1-5 | Questionnaire study only | all | yes | Confidence solving an unfamiliar problem (1-5; repeats Q19 Problem Solving) |
| `dev_practices` | Q28 | multi | any of `version_control`, `agile_scrum`, `code_reviews`, `testing`, `documentation`, `cicd`, `none` | Questionnaire study only | all | yes | Industry practices used (repeats Q18 Git/Testing/DevOps and Q29) |
| `improvement_areas` | Q30 | multi | any of `programming`, `database`, `cloud`, `cybersecurity`, `devops`, `ai_ml`, `communication`, `problem_solving`, `interview_skills`, `professional_experience` | Questionnaire study only | all | yes | Areas the graduate believes need improvement |
| `lacked_skills` | Q31 | multi | any of `programming_language`, `framework`, `cloud`, `database`, `git`, `communication`, `problem_solving`, `experience`, `certifications` | Questionnaire study only | all | yes | Skills employers or interviewers said were lacking |
| `explain_nontechnical` | Q32 | likert | 1-5 | Questionnaire study only | all | yes | Comfort explaining technical concepts to non-technical people (1-5; repeats Q19 Communication) |
| `team_confidence` | Q33 | likert | 1-5 | Questionnaire study only | all | yes | Confidence working in a development team (1-5; repeats Q19 Teamwork) |
| `group_projects` | Q34 | binary | `yes`, `no` | Questionnaire study only | all | yes | Participated in group software projects |
| `group_project_count` | Q34a | categorical | `1-2`, `3-5`, `5+` | Questionnaire study only | all | yes | Number of group software projects (if Q34 = yes) |
| `learning_strategies` | Q35 | multi | any of `online_courses`, `documentation`, `practice_projects`, `mentors`, `community`, `university`, `ai_tools` | Questionnaire study only | all | yes | How a missing skill is usually learned |
| `job_platforms` | Q36 | multi | any of `linkedin`, `topjobs`, `indeed`, `company_sites`, `university_career`, `agencies`, `other` | Questionnaire study only | all | yes | Platforms used to search for IT employment |
| `cv_update_freq` | Q37 | categorical | `never`, `yearly`, `few_months`, `every_application` | Questionnaire study only | all | yes | How often the CV is updated |
| `linkedin_profile` | Q38 | binary | `yes`, `no` | Questionnaire study only | all | yes | Has a LinkedIn profile |
| `linkedin_completeness` | Q38a | likert | 1-5 | Questionnaire study only | all | yes | LinkedIn profile completeness (1-5, if Q38 = yes) |
| `interview_feedback` | Q39 | binary | `yes`, `no` | Questionnaire study only | all | yes | Received feedback from employers after interviews |
| `feedback_areas` | Q39a | multi | any of `technical`, `communication`, `confidence`, `cultural_fit`, `experience`, `certifications`, `salary`, `other` | Questionnaire study only | all | yes | Areas mentioned in interview feedback (if Q39 = yes) |
| `self_employability` | Q40 | likert | 1-5 | Questionnaire study only | all | yes | Self-perceived employability (1-5) |
| `believed_factors` | Q41 | multi | any of `technical_skills`, `degree_class`, `internship`, `projects`, `certifications`, `communication`, `networking`, `interview_prep`, `university_reputation`, `industry_connections` | Questionnaire study only | all | yes | Factors believed to influence employability most (up to 5) |
| `university_prepared` | Q42 | likert | 1-5 | Questionnaire study only | all | yes | University prepared me for industry (1-5) |
| `biggest_challenge` | Q43 | categorical | `technical_skills`, `experience`, `confidence`, `industry_knowledge`, `job_opportunities`, `interviews`, `other` | Questionnaire study only | all | yes | Biggest challenge entering the IT job market |
| `explanation_importance` | Q44 | likert | 1-5 | Questionnaire study only | all | yes | Importance of understanding the reasons for a prediction (1-5) |
| `explanation_types` | Q45 | multi | any of `strengths`, `improvements`, `industry_comparison`, `learning_paths`, `similar_profiles` | Questionnaire study only | all | yes | Most useful explanation types |
| `would_use_system` | Q46 | likert | 1-5 | Questionnaire study only | all | yes | Would use an AI employability system before applying (1-5) |
| `comments` | Q47 | text | free text | Questionnaire study only | all | yes | Additional comments (optional) |

### Employer questionnaire

| Field | Item | Type | Valid values | Role | Applies to | Nullable | Description |
|---|---|---|---|---|---|---|---|
| `respondent_role` | Q3 | categorical | `hr_manager`, `recruiter`, `tech_lead`, `engineer_hiring`, `eng_manager`, `project_manager`, `cto_cio`, `other` | Questionnaire study only | all | yes | Respondent's role |
| `recruit_experience` | Q4 | categorical | `<1`, `1-3`, `4-6`, `7-10`, `10+` | Questionnaire study only | all | yes | Years of IT recruitment experience |
| `roles_recruited` | Q5 | multi | any of `software_engineer`, `qa`, `data_analyst`, `data_scientist`, `ai_ml_engineer`, `cloud`, `devops`, `cybersecurity`, `business_analyst`, `network_admin`, `ui_ux`, `other` | Questionnaire study only | all | yes | Entry-level IT roles recruited |
| `annual_hires` | Q6 | categorical | `0`, `1-5`, `6-10`, `11-25`, `25+` | Questionnaire study only | all | yes | Entry-level IT graduates recruited per year |
| `emp_english_importance` | Q9a | likert | 1-5 | Questionnaire study only | all | yes | Importance of English proficiency for entry-level IT graduates (1-5) |
| `future_tech` | Q10 | multi | any of `ai`, `ml`, `cloud`, `cybersecurity`, `data_engineering`, `devops`, `automation`, `blockchain`, `iot`, `software_engineering` | Questionnaire study only | all | yes | Technologies expected to grow in importance (3-5 years) |
| `skill_update_freq` | Q11 | categorical | `more_than_yearly`, `1-2_years`, `3-5_years`, `rarely` | Questionnaire study only | all | yes | How often required technical skills are updated |
| `tech_weaknesses` | Q12 | multi | any of `programming`, `se_practices`, `database`, `cloud`, `testing`, `cybersecurity`, `devops`, `ai_ml`, `data_analysis`, `other` | Questionnaire study only | all | yes | Technical weaknesses observed in fresh graduates |
| `soft_weaknesses` | Q13 | multi | any of `communication`, `teamwork`, `confidence`, `problem_solving`, `adaptability`, `professional_behaviour`, `time_management`, `attitude` | Questionnaire study only | all | yes | Soft-skill weaknesses observed in fresh graduates |
| `grad_alignment` | Q14 | likert | 1-5 | Questionnaire study only | all | yes | How well graduates meet expectations (1-5) |
| `uni_improvements` | Q15 | multi | any of `industry_projects`, `internships`, `modern_tech`, `soft_skills`, `career_prep`, `certifications`, `practical_dev` | Questionnaire study only | all | yes | Areas universities should improve |
| `ai_usefulness` | Q16 | likert | 1-5 | Questionnaire study only | all | yes | Usefulness of an AI employability system (1-5) |
| `emp_explanation_importance` | Q17 | likert | 1-5 | Questionnaire study only | all | yes | Importance of explanations from an AI system (1-5) |
| `ai_outputs` | Q18 | multi | any of `strengths`, `missing_skills`, `suitability_score`, `training_areas`, `job_roles`, `interview_prep` | Questionnaire study only | all | yes | Useful AI-generated outputs |
| `hiring_challenges` | Q19 | multi | any of `tech_skills`, `curriculum_gap`, `soft_skills`, `experience`, `salary_expectations`, `attrition`, `long_process`, `other` | Questionnaire study only | all | yes | Biggest challenges when hiring fresh graduates (up to 3) |
| `rank_technical_depth` | Q20.technical_depth | integer | 1-3 | Questionnaire study only | all | yes | Rank of 'Technical skill depth' among the top 3 (1 = most important) |
| `rank_portfolio` | Q20.portfolio | integer | 1-3 | Questionnaire study only | all | yes | Rank of 'Portfolio / project quality' among the top 3 (1 = most important) |
| `rank_interview` | Q20.interview | integer | 1-3 | Questionnaire study only | all | yes | Rank of 'Interview performance' among the top 3 (1 = most important) |
| `rank_certifications` | Q20.certifications | integer | 1-3 | Questionnaire study only | all | yes | Rank of 'Professional certifications' among the top 3 (1 = most important) |
| `rank_internship` | Q20.internship | integer | 1-3 | Questionnaire study only | all | yes | Rank of 'Internship experience' among the top 3 (1 = most important) |
| `rank_communication` | Q20.communication | integer | 1-3 | Questionnaire study only | all | yes | Rank of 'Communication skills' among the top 3 (1 = most important) |
| `rank_problem_solving` | Q20.problem_solving | integer | 1-3 | Questionnaire study only | all | yes | Rank of 'Demonstrated problem-solving ability' among the top 3 (1 = most important) |
| `rank_cultural_fit` | Q20.cultural_fit | integer | 1-3 | Questionnaire study only | all | yes | Rank of 'Cultural fit' among the top 3 (1 = most important) |
| `emp_comments` | Q21 | text | free text | Questionnaire study only | all | yes | Additional comments (optional) |

## 8. Job ads (collected manually) - raw fields

| Field | Item | Type | Valid values | Role | Applies to | Nullable | Description |
|---|---|---|---|---|---|---|---|
| `ad_id` | manual | id | free text | Identifier | all | no | Row identifier |
| `title` | manual | text | free text | Descriptive only | all | no | Advertised job title |
| `company` | manual | text | free text | Descriptive only | all | no | Hiring company |
| `date` | manual | month | YYYY-MM | Descriptive only | all | no | Date the ad was collected |
| `source` | manual | categorical | `linkedin`, `topjobs`, `company_site`, `other` | Descriptive only | all | no | Where the ad was found |
| `skills` | manual | text | free text | Descriptive only | all | no | Semicolon-separated SKILL_VOCABULARY keys mentioned in the ad |

## 9. Outcome-construction constants

- `JOB_BEFORE_GRADUATION_MONTHS` = 0.5
- `PREDICTION_HORIZON_MONTHS` = 24
- `EPV_WARNING_THRESHOLD` = 10
