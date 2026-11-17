# Skillfolio demo and teacher guide

**One line:** Skillfolio compares your skills with tech roles, tests selected skills, and simulates what you could learn next.

**Run it:** `cd ~/CareerTwin && SKILLFOLIO_LOCAL_HISTORY=1 .venv/bin/streamlit run app.py`
**Run the tests:** `.venv/bin/python tests/test_careertwin.py` (29 checks, all pass)

## The data (real, not synthetic)

- Source: Kaggle, "1.3M LinkedIn Jobs & Skills (2024)".
- We kept tech postings and labelled each with a role from its job title (13 roles).
- Duplicates were removed so the same ad cannot sit in both train and test.
- Skill-list model: **27,219 postings, 500 skills.** Text models: **22,211 full job descriptions.**
- All training ran on Kaggle: `training/train.py` (CPU) and `training_deep/train_deep.py` (GPU).

## Three trained models

| Model | Job | Trained on | Result |
|---|---|---|---|
| Logistic regression (softmax) | skill list → role | 21,775 postings | 77.3% accuracy, 95.4% top-3, ROC-AUC 0.972 |
| DistilBERT role model | resume or JD text → role | 17,768 job descriptions | 81.6% accuracy, 95.3% top-3 |
| DistilBERT skill tagger (NER) | marks the skill words in any text | 24,203 text windows | precision 56%, recall 54%, F1 0.55 |

Both DistilBERT models have 66 million weights. They were shrunk from 268 MB to 67 MB each by 8-bit quantisation so they run on a laptop CPU in about 30 ms per document.

## Where to demo each feature

| Tab | What it does | Algorithm | Module |
|---|---|---|---|
| Resume | Highlights skills in your text, guesses your role, shows which sentences help | Transformers (BERT), NLP, occlusion test | V |
| Role fit | Probability for each role, and the weight of each of your skills | Logistic regression, softmax | I |
| Explore → Skill map | 3D star map where related skills sit together | Co-occurrence, SVD, t-SNE, K-means | IV |
| Explore → Learning path | Finds the best new skills, orders them, replays the search | Genetic algorithm, cosine similarity | V |
| Explore → Job description | Paste a job ad: what it asks for, what you have, what you lack | Both transformers | V |
| Explore → Similar jobs | Real postings most like you | K-nearest neighbours | III |
| Skill test | Adaptive test, mastery report, and retake history | IRT, Bayes, BKT | I, III |
| Coding profile | Public profiles, rating estimates, and practice suggestions | Regression, K-means | I, IV |
| Model lab | 8 classical models compared, deep vs classical, live audit | Modules I, II, III; MLflow and deployment (V) | all |

## Likely viva questions

**Is the model real or hard-coded?**
Open Model lab and press "Re-test". The app recomputes accuracy on 5,444 postings it never trained on and gets 77.3%, the same as Kaggle. Then it scrambles which column means which skill and accuracy falls to 13% (pure guessing is 7.7%). A hard-coded answer would not change.

**How is the percentage calculated?**
Each skill has a weight for each role. We add the weights of your skills and softmax turns the 13 totals into probabilities.

**How does the genetic algorithm work here?**
- Chromosome: a set of k skills you do not have yet.
- Fitness: probability of the target role after adding them, minus a penalty for skills far from what you know.
- Tournament selection of 3, crossover takes k skills from both parents, 30% mutation, the best 2 always survive.
- 60 sets per generation, 40 generations. In the test it beat 2,000 random skill sets (0.557 vs 0.183).

**What is "learning effort"?**
Skills that appear together in job ads get similar vectors. Effort = 1 minus cosine similarity to your nearest known skill.

**How does the skill tagger know what a skill is?**
We fine-tuned DistilBERT for token classification. Training labels came from the dataset: wherever a posting's listed skill appeared in its text, those words were tagged. The model learned the pattern, so it also tags tools that are not in our 500-skill list (in the test it found "Qdrant").

**How do you know which sentence helps?**
Occlusion. Delete one sentence, re-run the model, measure the drop in the role's probability.

**Did you stop the text model from cheating?**
Yes. Job descriptions often contain the job title. We deleted the title and title-like phrases from every description before training.

**Why did the simple model beat the transformer?**
TF-IDF + logistic regression scored 83.4% on job descriptions; DistilBERT scored 81.6%. The transformer only reads the first 320 tokens, while TF-IDF sees the whole document. We report this as it is. We still use DistilBERT in the app because it works on short resume text and needs no 50,000-word vocabulary.

**Why logistic regression when Random Forest scored higher on skills?**
Random Forest got 80.0% but 99.9% on its own training data, so it memorises. Logistic regression got 77.3% and every weight can be shown on screen.

## Classical model comparison

| Model | Test accuracy | Train accuracy |
|---|---|---|
| Random Forest | 80.0% | 99.9% |
| Gradient Boosting | 79.8% | 92.7% |
| Neural Network (MLP) | 78.1% | 83.4% |
| Linear SVM | 77.5% | 81.6% |
| Logistic Regression | 77.1% | 82.8% |
| KNN | 76.0% | 79.0% |
| Naive Bayes | 70.4% | 71.6% |
| Decision Tree | 63.2% | 79.0% |

## Honest limits (say these before the teacher asks)

- Role labels come from job titles, so they are noisy, and roles overlap in real life.
- The score means "your profile looks like postings for this role". It is not a promise of placement.
- The skill tagger misses about half the skills measured against the dataset's own labels, and 8-bit compression cost it about 5 points of F1. It missed "scikit-learn" in our sample resume.
- The text model was trained on job descriptions, not resumes, so resume results are a transfer.
- The data is LinkedIn 2024 and leans towards US postings.
- The 3D view and web fonts load from the internet. The flat skill map works offline.


---

# Skill test and coding profile

**Run the new tests:** `.venv/bin/python tests/test_new.py` (20 checks; add `--online` to also check the live LeetCode, Codeforces and GitHub fetches)

## Skill test (tab "Skill test")

You pick one of 30 skills (Docker, MLOps, Python, SQL, DSA...) and answer 5 to 10 multiple-choice questions. The question bank has 540 questions: 18 per skill, 6 easy, 6 medium, 6 hard, split over 3 subtopics, each with a short explanation.

How it scores you, in plain words:

| Step | What happens | Syllabus |
|---|---|---|
| Belief about you | Starts as a bell curve over "ability". The chart on the right shows it. | Probability distribution (I) |
| After each answer | Bayes' theorem: new belief = old belief × chance of that answer at each ability. The curve narrows. | Bayes theorem (III) |
| Chance of a right answer | Item Response Theory: a sigmoid of (ability − difficulty), plus a small chance of guessing and of slipping | Sigmoid, logistic regression (I) |
| Next question | The difficulty that gives the most information at your current estimate | Adaptive testing |
| When to stop | When the belief is narrow enough (spread < 0.55), after at least 5 and at most 10 questions | — |
| Subtopic bars | Bayesian Knowledge Tracing, a hidden Markov model of "learned / not learned" | Bayesian learning, HMM (III) |
| "Not sure" button | A correct answer marked "not sure" counts half, because it may be a guess | — |
| Fading | Mastery shrinks over days along a forgetting curve | — |
| Role fit | A switch makes tested skills count by their score (Docker 40% counts as 0.4) | — |

**Training and evaluation.** Riiid contains 99,271,300 answers from 393,656 students. The Kaggle run sampled 60,000 students, trained on 48,000 and evaluated on 671,207 answers from the other 12,000. The ASSISTments run failed, so the results below come from Riiid.

| Model (predict the student's next answer) | AUC | Accuracy |
|---|---|---|
| Item average (baseline) | 0.728 | 67.7% |
| IRT 2PL | 0.757 | 69.4% |
| **IRT 4PL (guess + slip)** | **0.757** | **69.5%** |
| IRT with only 3 difficulty levels (how our app works) | 0.726 | 67.3% |
| Bayesian Knowledge Tracing | 0.646 | 62.2% |
| Deep Knowledge Tracing (LSTM, 8.4M weights) | 0.729 | 67.8% |

Fitted values the app uses: guess 0.10, slip 0.00, discrimination 1.43, difficulty levels −1.01 / −0.28 / +0.53, BKT (prior 0.66, learn 0.46, guess 0.29, slip 0.28), forgetting half-life about 4 days.

In simulation (`tests/test_new.py`), the estimated ability correlated 0.94 with simulated ability. Tests averaged 8.6 questions. The question-bank checks cover structure and answer uniqueness. Review the answer keys for the skills you plan to demo. Hosted demo retakes share history within a session. The local run command above enables test history on disk.

**Viva questions**
- *Why IRT and not the LSTM?* IRT scored best (AUC 0.757 vs 0.729), it works on brand-new questions, and every number in it can be explained. The LSTM needs past answers to the same questions, which our new bank does not have.
- *Why did BKT score lowest?* It only knows "learned or not" per skill, and ignores that some questions are harder than others. We still use it for subtopic bars because it is simple to explain.
- *What is a real limitation?* Our bank has only 3 difficulty levels. On real data, IRT with only 3 levels drops to the baseline (0.726), and adaptive ordering barely beats random order with 3 levels (error 0.51 vs 0.53 after 10 questions), while with 1,500 real difficulty levels the error falls to 0.25. More question levels would make the test sharper.

## Coding profile (tab "Coding profile")

Enter a LeetCode, Codeforces and/or GitHub username. All three are public data (LeetCode's API is unofficial and could change).

**Codeforces models** (`training_cf/train_cf.py`, Kaggle CPU): we collected 1,348 real coders through the official Codeforces API, evenly across rating bands, out of 44,751 active ones. Features: how many problems solved at each difficulty, topics solved, solve ratio. Task: predict rating.

| Model | MAE | R² |
|---|---|---|
| Gradient Boosting | 178 | 0.833 |
| Neural Network (MLP), **deployed** | 179 | 0.830 |
| Random Forest | 179 | 0.834 (but train MAE 89: overfits) |
| Ridge (L2) | 185 | 0.817 |
| Linear Regression | 186 | 0.816 |
| KNN | 185 | 0.817 |
| Lasso (L1) | 190 | 0.812 |
| Decision Tree | 225 | 0.706 |
| Elastic Net | 227 | 0.743 |
| Guessing the average | 490 | 0 |

- **"Model says" vs your rating:** if your practice predicts a higher rating than you have, you are under-performing in contests; lower means you do better in contests than your practice suggests.
- **Coding style:** K-means on each coder's topic mix, k = 3 chosen by silhouette: "two pointers / greedy / binary search", "trees / shortest paths / divide and conquer", "strings / implementation / math".
- **Next-level gap:** median topic counts of coders one band (200 points) above you, minus yours. Then it recommends unsolved problems 100–300 points above your rating in your weakest topics, most-solved first.
- **LeetCode:** solved by difficulty, "beats %" from LeetCode itself, contest rating, top topics, interview topics with fewer than 10 solved.
- **GitHub:** languages in your recent repos; a button adds them to your profile as proven skills (a Dockerfile counts as Docker).
- **DSA readiness:** the higher of your Codeforces percentile and a simple LeetCode score ((easy + 2.5 × medium + 5 × hard) / 4, capped at 100). This one is a rule, not a trained model; say so.

**Limitation:** the rating model shrinks extremes towards the middle. For the world's top coder (rating 3408) it predicts 2794.
