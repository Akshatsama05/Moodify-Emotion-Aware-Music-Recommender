🎧 Moodify --- Emotion-Aware Music Recommender



An NLP and machine-learning music recommendation system that

converts a user's natural-language feeling into a six-emotion playlist

direction and ranks songs from a multilingual catalogue.



Moodify combines TF-IDF text representation, a balanced Linear SVM

emotion classifier, metadata/content-based song-emotion relevance,

TF-IDF query similarity, popularity ranking, language-aware cues, and a

Streamlit interface.



The current repository is deliberately transparent about its evidence

boundary: the bundled song catalogue contains metadata, but does not

contain human song-emotion labels, lyrics, or audio-derived emotion

features. Therefore, catalogue emotion values are used as

metadata/content relevance signals, not as ground-truth claims about

how a song objectively feels.



📌 Project Overview



Moodify addresses a simple user problem:



"I know how I feel, but I don't know what music fits that

feeling."



Traditional playlist systems often depend on listening history, explicit

song ratings, or fixed genre/category selections. Moodify explores a

different interaction: the user describes their current feeling in

ordinary language, and the system translates that text into one of six

emotion classes before ranking relevant songs.



Production emotion classes



Moodify preserves the original six classes from the supervised emotion

dataset:



Class           Meaning



😢 sadness    Sadness, grief, loneliness, heartbreak

😊 joy        Happiness, celebration, excitement

❤️ love       Romance, affection, attachment

😠 anger      Anger, frustration, hostility, conflict

😨 fear       Fear, anxiety, danger, uncertainty

😲 surprise   Surprise, shock, amazement, unexpected events



No three-mood mapping is used in the production path.



🎯 Problem Statement



Emotion-aware recommendation has two distinct technical problems:



Understand the user's text.

A user may write "I had a rough day and just want something quiet

and emotional." The system must infer the dominant emotion from

natural language.



Find songs that fit the inferred emotion.

The catalogue does not provide reliable human-labelled song

emotions. A recommendation system therefore cannot honestly treat

metadata heuristics as psychological ground truth.



Moodify separates these problems instead of hiding the second limitation

behind a misleading label.



The supervised NLP model is evaluated on a held-out

emotion-classification test set. The song catalogue uses a separate,

explicitly documented metadata/content relevance layer.



💡 Solution



The production flow is:



User mood description

&#x20;       │

&#x20;       ▼

TF-IDF vectorization

&#x20;       │

&#x20;       ▼

Balanced Linear SVM

&#x20;       │

&#x20;       ├── predicted emotion

&#x20;       └── normalized decision-score weights

&#x20;       │

&#x20;       ▼

Selected six-emotion catalogue relevance

&#x20;       │

&#x20;       ├── lexical / phrase evidence

&#x20;       ├── TF-IDF similarity to emotion prototypes

&#x20;       └── optional weak genre cues

&#x20;       │

&#x20;       ▼

User-query TF-IDF similarity

&#x20;       │

&#x20;       ▼

Popularity as secondary signal

&#x20;       │

&#x20;       ▼

Evidence-gated ranking + duplicate removal

&#x20;       │

&#x20;       ▼

Top-10 multilingual recommendations

&#x20;       │

&#x20;       ▼

Streamlit UI + Spotify links + artwork



The architecture is intentionally modular so that future validated

lyrics/audio models or human song-emotion annotations can be integrated

without rewriting the core recommender.



✨ Key Capabilities



🧠 Six-class natural-language emotion classification



📚 TF-IDF unigrams + bigrams



⚖️ Class-balanced Linear SVM



📊 Held-out evaluation with accuracy, balanced accuracy, macro-F1,

classification report and confusion matrix



🎵 2,878-song catalogue



🌍 English, Hindi and Punjabi catalogue coverage



🔤 English + Romanized Hindi/Punjabi emotion cues



🧩 Phrase-level emotion evidence



🧮 Six-way catalogue emotion relevance profiles



🔎 Query-to-song TF-IDF similarity



📈 Popularity as a secondary ranking signal



🛡️ Conservative emotion-evidence gate



🔁 Duplicate title/artist removal



🔀 Mood-constrained shuffle



🖼️ Album artwork and Spotify links



🖥️ Streamlit application



☁️ Google Colab end-to-end workflow



🔬 Human-validation workflow without requiring a Google Form



🎚️ Optional lyrics/audio multimodal integration layer



✅ Project regression/health checks



🔍 Explicit limitation and evidence documentation



🏗️ System Architecture



1\. User-text emotion layer



The user enters a short natural-language description.



Example:



I feel lonely and nostalgic and want something emotional.



The production model converts the text into TF-IDF features and applies

a Linear SVM trained on six emotion classes.



The classifier returns:



a predicted emotion, and



six normalized, probability-like weights derived from the SVM

decision scores.



These normalized values are used for ranking only. They are not

claimed to be calibrated probabilities.



2\. Catalogue emotion-relevance layer



The supplied catalogue does not provide validated song-emotion labels.



Moodify therefore constructs six independent relevance signals from

available metadata:



song title



album



genre



language



Artist names are deliberately excluded from emotion scoring to reduce

artist-identity leakage.



Lexical evidence



Emotion-specific vocabularies contain English and multilingual/Romanized

cues.



For example, the love vocabulary contains terms such as:



love, romance, affection, relationship,

ishq, pyaar, pyar, mohabbat, prem,

sanam, mahi, yaara



The system also contains phrase-level cues because a multi-word

expression can be much more informative than an isolated token.



Semantic/prototype similarity



Moodify builds six emotion prototypes and computes TF-IDF cosine

similarity between catalogue metadata text and those prototypes.



The prototype vocabulary includes multilingual cues so that

Hindi/Punjabi catalogue metadata is not evaluated exclusively against

English emotion words.



Weak genre prior



Genre information can provide a small additional signal for selected

emotions. This is deliberately low-weight because broad genre categories

are not reliable emotion ground truth.



📊 Dataset \& Data Preparation



Supervised emotion dataset



The project preserves a fixed:



16,000-row training split



2,000-row validation split



2,000-row held-out test split



Total:



20,000 text examples



The production model is fit using the training + validation data after

model selection, while the 2,000-row test set remains the held-out

evaluation set.



Class distribution in the held-out test set



Emotion       Test support



sadness                581

joy                    695

love                   159

anger                  275

fear                   224

surprise                66

Total        2,000



The class imbalance is one reason the final classifier uses:



class\_weight = "balanced"



Song catalogue



The bundled catalogue contains:



2,878 tracks



Language coverage:



Language         Tracks



English             899

Hindi             1,000

Punjabi             979

Total     2,878



The raw catalogue includes fields such as:



id



song\_name



singer



album



release\_date



cover\_image



spotify\_url



popularity



genre



language



language\_evidence



source



The preprocessing layer also creates:



artist\_text



release\_year



text



recommendation\_text



Why two text fields?



text retains artist identity for general catalogue text/search use.



recommendation\_text deliberately excludes artist names:



song\_name + album + genre + language



This reduces the chance that an artist's identity or historically

associated songs dominate mood matching.



🧠 Emotion Classification



TF-IDF representation



The production classifier uses:



lowercase=True

ngram\_range=(1, 2)

min\_df=2

max\_features=12000

sublinear\_tf=True



This means the model can represent both individual words and two-word

phrases.



Examples:



lonely

heartbreak

very happy

broken heart



TF-IDF is useful here because emotion-bearing words and phrases can

provide strong discriminative information without requiring a large

transformer-based architecture.



Model comparison



The reusable modelling module evaluates three classical NLP baselines:



Multinomial Naive Bayes



Logistic Regression



Linear SVM



All use the same TF-IDF representation so that model comparison is

reasonably controlled.



The production artifact is a scikit-learn Pipeline containing:



TfidfVectorizer → LinearSVC



with:



class\_weight="balanced"

C=1.0

random\_state=42

max\_iter=1000



📈 Model Evaluation



The final Linear SVM is evaluated once on the held-out 2,000-row test

set.



Overall performance



Metric                     Score



Accuracy              89.05%

Balanced Accuracy     85.26%

Macro-F1              84.38%



Per-emotion results



Emotion      Precision   Recall       F1



sadness         93.51%   91.74%   92.62%

joy             92.94%   90.94%   91.93%

love            74.46%   86.16%   79.88%

anger           86.27%   89.09%   87.66%

fear            88.26%   83.93%   86.04%

surprise        66.67%   69.70%   68.15%



Interpretation



The overall accuracy is strong, but the macro-F1 and per-class scores

are more informative because the classes are imbalanced.



In particular:



sadness and joy are the strongest classes.



love, fear, and anger show useful but imperfect separation.



surprise is the weakest class, partly reflecting its much smaller

test support.



These metrics measure user-text emotion classification. They do

not measure whether a song objectively expresses an emotion.



🎵 Catalogue Emotion Profiling



The catalogue-side emotion layer creates a complete six-way profile for

every track.



For each song, the processed catalogue can contain:



emotion\_sadness

emotion\_joy

emotion\_love

emotion\_anger

emotion\_fear

emotion\_surprise



and normalized profile fields:



emotion\_prob\_sadness

emotion\_prob\_joy

emotion\_prob\_love

emotion\_prob\_anger

emotion\_prob\_fear

emotion\_prob\_surprise



These are relevance estimates, not human-labelled probabilities.



A conservative display label is also created:



emotion\_label

emotion\_confidence

emotion\_label\_source



If available metadata does not provide enough evidence, the track can

remain:



unclassified



This is an intentional design choice.



Why unclassified matters



An earlier implementation gave every emotion a universal baseline and

then selected the first maximum. That could turn a song with no emotion

evidence into sadness.



The corrected implementation removes that tie bias.



The current catalogue distribution is:



Label            Count



sadness             20

joy                 41

love               167

anger               12

fear                 9

surprise             6

unclassified     2,623



The large unclassified group is not treated as a failure to hide. It

reflects the actual evidence available from metadata.



🧮 Recommendation Engine



The recommender does not simply filter songs by emotion\_label.



Instead, it ranks the catalogue using the complete emotion profile.



For a selected emotion:



emotion\_score

&#x20;       +

query similarity

&#x20;       +

popularity



The current ranking formula is:



Rank Score =

&#x20;   0.70 × Emotion Score

&#x20; + 0.20 × TF-IDF Similarity

&#x20; + 0.10 × Popularity Score



Where:



70% emotion relevance keeps the recommendation mood-driven.



20% query similarity personalizes the ranking around the actual

user description.



10% popularity acts as a secondary quality/tie-break signal.



Popularity is normalized from the catalogue's 0--100 scale.



🛡️ Top-10 Evidence Gate



Moodify applies a conservative:



MIN\_EMOTION\_EVIDENCE = 0.10



threshold.



The recommender first ranks tracks with at least this level of raw

emotion evidence.



If there are at least 10 strong candidates:



Top 10 = strongest evidence candidates



If there are fewer than 10:



strong candidates

&#x20;       +

best remaining candidates

&#x20;       ↓

complete Top 10 when possible



This is a deliberate precision/recall trade-off.



It prevents low-evidence tracks from outranking clearly supported tracks

while still allowing the application to return a useful Top-10 list when

the catalogue is sparse.



This gate does not create ground-truth song-emotion labels.



🔎 TF-IDF Query Similarity



The recommendation layer builds a separate TF-IDF representation over:



recommendation\_text



The user's original mood description is transformed using the same

vectorizer.



Cosine similarity then measures how closely the query text relates to

each song's metadata/content representation.



This signal is secondary to emotion relevance.



That means the system is designed to prefer:



emotion fit first, textual relevance second, popularity third.



🔁 Duplicate Handling



Recommendations remove duplicate:



song\_name + artist\_text



combinations.



This prevents repeated catalogue records from consuming multiple

recommendation slots while still allowing different songs with the same

title to remain when their artists differ.



🔀 Shuffle Mode



Shuffle does not sample randomly from the entire catalogue.



The application first creates a ranked candidate set for the selected

emotion and then shuffles that candidate set.



This prevents a "shuffle" action from returning songs unrelated to the

chosen mood.



🌍 Multilingual Support



Moodify's current catalogue covers:



English



Hindi



Punjabi



The emotion-relevance vocabulary also includes Romanized Indian-language

cues.



Examples include:



udaas

dukh

tanhai

judai

khushi

masti

ishq

pyaar

mohabbat

gussa

nafrat

darr

khauf

hairan

achanak



Language can be selected in the Streamlit sidebar:



All

English

Hindi

Punjabi



The language selector changes the recommendation pool; it does not

retrain or modify the emotion classifier.



🖥️ Streamlit Application



The production UI is implemented in app.py.



User flow



1\. Describe your feeling

&#x20;           ↓

2\. Click "Analyze mood"

&#x20;           ↓

3\. View model prediction / selected emotion

&#x20;           ↓

4\. Select language

&#x20;           ↓

5\. Receive Top-10 recommendations

&#x20;           ↓

6\. Open individual songs on Spotify



UI features



Responsive Streamlit layout



Mood input text area



Example prompts



Six-emotion selector



Language selector



Mood summary



Model prediction display



Selected emotion score



Top-10 recommendation cards



Album artwork



Artist and album information



Popularity display



Spotify links



Shuffle mode



NLP explanation panel



Project limitation notes



🔬 Explainability



Moodify includes a reusable explain\_prediction() function in

src/recommender.py.



For a Linear SVM prediction, it can inspect the trained TF-IDF

vocabulary and classifier coefficients to identify terms contributing

most strongly to the selected class.



This provides a lightweight, model-native explanation mechanism without

pretending that the classical classifier is inherently interpretable.



🧩 Multimodal Architecture



The repository contains an optional multimodal integration layer in:



src/multimodal.py



The current bundled project does not fabricate or include

lyrics/audio-derived emotion features.



Instead, the architecture supports future feature tables containing

six-way emotion profiles such as:



lyrics\_prob\_sadness

lyrics\_prob\_joy

...

audio\_prob\_sadness

audio\_prob\_joy

...



Safe joining



Optional feature tables must contain the exact catalogue:



id



and are joined using that ID only.



Title-only matching is intentionally rejected because:



remixes can share titles



different artists can share titles



different versions can exist



duplicate titles can collide



Fusion weights



When legitimate optional modalities are available, the integration layer

is designed around:



Metadata = 0.45

Lyrics   = 0.30

Audio    = 0.25



Missing modalities are excluded and the available weights are

renormalized.



If no optional modality exists, Moodify keeps the metadata-only profile.



👥 Human Validation Framework



A major limitation of the current catalogue is the absence of human

song-emotion ground truth.



Instead of pretending the metadata labels are correct, the repository

includes a reproducible local validation workflow.



Included validation queue



reports/human\_validation/annotation\_queue\_300.csv



The queue contains:



100 English songs



100 Hindi songs



100 Punjabi songs



A local annotation interface is provided by:



tools/annotation\_app.py



Three independent annotators per song are recommended.



Evaluation



After real annotations are supplied:



tools/evaluate\_human\_validation.py



can compute:



catalogue emotion accuracy



balanced accuracy



macro-F1



confusion matrix



mean majority agreement



Fleiss' kappa



recommendation Precision@5



recommendation Precision@10



NDCG@5



NDCG@10



artist diversity



Important evidence rule



The evaluator refuses to fabricate human metrics when annotations are

absent.



Therefore, the current repository is:



human-validation ready, but not human-validated yet.



🗂️ Project Structure



Moodify/

│

├── app.py

├── run\_local.py

├── run\_local.bat

├── run\_ngrok.bat

├── requirements.txt

├── validate\_project.py

│

├── README.md

├── LOCAL\_RUN\_GUIDE.md

├── DEPENDENCY\_MAP.md

├── FINAL\_FIX\_REPORT.md

├── final\_quality\_audit.md

├── integration\_verification.md

├── validation\_summary.md

├── vad\_mapping\_experiment.md

│

├── Moodify\_Master\_Colab.ipynb

├── Moodify\_Master\_Colab\_executed.ipynb

│

├── emotion\_train.csv

├── emotion\_validation.csv

├── emotion\_test.csv

├── emotion\_model\_metrics.json

├── emotion\_model\_report.md

│

├── data/

│   ├── raw/

│   │   └── spotify\_metadata\_catalogue.csv

│   │

│   ├── processed/

│   │   └── labelled\_catalogue.csv

│   │

│   └── external/

│       └── README.md

│

├── models/

│   └── moodify\_model.joblib

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── data.py

│   ├── emotion\_data.py

│   ├── labeling.py

│   ├── modeling.py

│   ├── multimodal.py

│   └── recommender.py

│

├── notebooks/

│   ├── Moodify\_End\_to\_End.ipynb

│   ├── Moodify\_End\_to\_End\_executed.ipynb

│   └── Streamlit\_Colab\_Launcher.py

│

├── tools/

│   ├── annotation\_app.py

│   ├── create\_annotation\_queue.py

│   ├── evaluate\_human\_validation.py

│   └── prepare\_multimodal\_features.py

│

└── reports/

&#x20;   ├── LIMITATION\_STATUS.md

&#x20;   ├── catalogue\_emotion\_recommendation\_audit.csv

&#x20;   ├── catalogue\_emotion\_score\_summary.csv

&#x20;   │

&#x20;   ├── figures/

&#x20;   │   ├── confusion\_matrix.png

&#x20;   │   ├── eda\_overview.png

&#x20;   │   ├── model\_comparison.png

&#x20;   │   └── weak\_label\_distribution.png

&#x20;   │

&#x20;   └── human\_validation/

&#x20;       ├── annotation\_queue\_300.csv

&#x20;       └── ANNOTATION\_INSTRUCTIONS.md



Important modules



src/emotion\_data.py



Defines the canonical six production emotions and mappings between

numeric labels and emotion names.



src/data.py



Handles catalogue loading, required-column validation, cleaning, date

conversion, artist-list parsing, duplicate removal, and recommendation

text preparation.



src/modeling.py



Contains reusable model construction, model comparison, train/test

splitting and evaluation helpers.



src/labeling.py



Implements metadata/content-based emotion relevance using:



lexical cues



phrase cues



multilingual prototypes



TF-IDF cosine similarity



weak genre cues



six-way normalization



conservative unclassified labelling



src/recommender.py



Contains:



emotion prediction



SVM decision-score normalization



recommendation ranking



evidence gating



duplicate removal



model-term explanation



src/multimodal.py



Provides the optional lyrics/audio feature join and six-way profile

fusion layer.



app.py



Contains the production Streamlit interface.



tools/annotation\_app.py



Local interface for collecting human song-emotion annotations.



tools/evaluate\_human\_validation.py



Evaluates real human annotations and recommendation quality without

fabricating missing metrics.



tools/prepare\_multimodal\_features.py



Validates and merges externally supplied lyrics/audio emotion profiles.



validate\_project.py



Runs regression checks for the project, including model classes,

catalogue size, emotion score columns and recommendation smoke tests.



⚙️ Installation



Requirements



Python 3.10+ recommended



pip



Git (optional, for cloning)



Internet connection for installing Python dependencies and opening

external Spotify links



Install dependencies:



pip install -r requirements.txt



🚀 Running Locally



The simplest option is:



streamlit run app.py



Then open:



http://127.0.0.1:8501



or:



http://localhost:8501



Recommended launcher



The repository also provides:



python run\_local.py local



On Windows:



run\_local.bat



The launcher:



installs/checks dependencies



starts Streamlit



performs a local health check



reports the local URL



No Cloudflare tunnel or authentication is required for the normal local

mode.



🌐 Optional ngrok Mode



An optional ngrok mode is available:



python run\_local.py ngrok



It requires:



NGROK\_AUTHTOKEN



to be configured in the environment.



This mode is optional. The normal application works locally without

ngrok.



☁️ Google Colab



The repository includes:



Moodify\_Master\_Colab.ipynb



and an executed version:



Moodify\_Master\_Colab\_executed.ipynb



The master notebook follows the same end-to-end architecture:



1\. Load six-class emotion data

2\. Train/evaluate the six-class model

3\. Prepare catalogue emotion relevance

4\. Run shared recommendation logic

5\. Save/verify the model artifact

6\. Launch Streamlit

7\. Perform health checks

8\. Optionally create a temporary public tunnel



The notebook is intended as a reproducible project workflow rather than

a separate implementation of the recommender.



🧪 Validation \& Regression Checks



Run:



python validate\_project.py



The project validation layer checks important production assumptions,

including:



model availability



six-class model compatibility



catalogue row count



six emotion score columns



recommendation smoke tests



normalized SVM ranking scores



documented limitation status



The repository also contains generated evaluation artifacts under:



reports/figures/



including:



model comparison



confusion matrix



EDA overview



weak-label distribution



📋 Reproducibility



The project stores the trained model artifact:



models/moodify\_model.joblib



The artifact is a scikit-learn pipeline containing:



TF-IDF Vectorizer

&#x20;       ↓

LinearSVC



The repository also includes the train, validation and test CSV files

used by the supervised modelling workflow.



This makes the project more reproducible than a repository containing

only an application screenshot or a notebook without the trained

artifact.



⚠️ Limitations



1\. Song emotion is not ground truth



The largest limitation is the song catalogue's lack of validated emotion

labels.



The current catalogue contains metadata, but not:



human song-emotion annotations



bundled lyrics



validated audio-emotion predictions



Therefore:



Moodify is an emotion-aware metadata/content recommender, not a

ground-truth music-emotion classifier.



2\. Metadata cannot fully capture musical emotion



Song titles and metadata cannot reliably represent:



melody



harmony



rhythm



vocal delivery



instrumentation



production style



sarcasm



lyrical context



personal associations



A song can have a happy title while sounding melancholic, or the

reverse.



3\. Language nuance



Romanized Hindi and Punjabi text is supported through explicit

lexical/prototype cues, but this does not provide full semantic

understanding of all dialects, slang, spelling variants or culturally

specific expressions.



4\. SVM scores are not calibrated probabilities



The application converts Linear SVM decision scores into normalized

softmax-like weights for ranking.



These values should be interpreted as:



probability-like ranking weights



not statistically calibrated probabilities.



5\. Human validation is pending



The repository contains the complete annotation/evaluation machinery,

but meaningful human song-emotion metrics require real independent

annotations.



No human validation scores are claimed until those annotations exist.



🧭 Responsible Interpretation



Moodify should be presented as:



An NLP-driven, emotion-aware music recommendation prototype using

six-class user-text emotion classification and metadata/content-based

song relevance.



It should not be presented as:



a psychological assessment tool



a system that objectively knows a person's emotional state



a human-validated music-emotion classifier



a clinically meaningful emotion detector



a ground-truth audio emotion recognition model



The distinction between prediction, relevance estimation, and

ground truth is an intentional engineering decision in this project.



🔮 Future Improvements



The architecture is designed to support several evidence-based upgrades.



1\. Human-labelled song-emotion dataset



Collect independent annotations for a representative subset of the

catalogue and evaluate:



agreement



macro-F1



balanced accuracy



Precision@K



NDCG@K



artist diversity



2\. Lyrics-based emotion modelling



Integrate a documented lyrics emotion model through the existing

id-based multimodal interface.



3\. Audio-based emotion modelling



Add validated audio-derived six-way emotion profiles using features such

as:



tempo



energy



spectral characteristics



rhythm



learned audio embeddings



4\. Multimodal fusion



Combine:



metadata + lyrics + audio



rather than relying primarily on metadata.



5\. Better multilingual NLP



Move beyond keyword/prototype cues toward multilingual transformer

representations while preserving the current six-class production

taxonomy.



6\. Calibrated uncertainty



Evaluate calibration methods for the user-text classifier instead of

treating raw SVM decision scores as confidence.



7\. Personalization



Incorporate user feedback such as:



like

skip

save

replay



to learn individual preferences while retaining emotion relevance.



8\. Ranking evaluation



Once human relevance labels exist, optimize the recommender directly

for:



Precision@K

NDCG@K

Recall@K

diversity

novelty



rather than relying primarily on heuristic ranking weights.



🧪 Technical Highlights



This project demonstrates practical ML engineering beyond simply

training a classifier.



Machine Learning



Multi-class text classification



TF-IDF feature engineering



Naive Bayes / Logistic Regression / Linear SVM comparison



Class balancing



Held-out evaluation



Confusion-matrix analysis



NLP



Unigrams and bigrams



Lexical emotion cues



Phrase-level matching



Multilingual/Romanized vocabulary



TF-IDF cosine similarity



Emotion prototypes



Recommendation Systems



Content-based ranking



Emotion-conditioned retrieval



Weighted ranking function



Popularity normalization



Evidence gating



Duplicate suppression



Constrained shuffle



Software Engineering



Modular src/ package



Reusable model/recommender functions



Persisted model artifact



Streamlit frontend



Local launcher



Colab workflow



Validation scripts



Explicit limitation tracking



Optional multimodal integration



Human annotation workflow



Responsible ML



No fabricated human labels



No fabricated lyrics/audio features



Explicit ground-truth boundary



Conservative unclassified state



Safe multimodal joins using song IDs



No claim that SVM scores are calibrated probabilities



📊 Current Project Snapshot



Component                              Current status



User-text emotion classifier           ✅ Implemented

Six production emotions                ✅ Implemented

TF-IDF + Linear SVM                    ✅ Implemented

Held-out test evaluation               ✅ Completed

Accuracy                               89.05%

Balanced accuracy                      85.26%

Macro-F1                               84.38%

Song catalogue                         2,878 tracks

Languages                              English, Hindi, Punjabi

Metadata emotion relevance             ✅ Implemented

Evidence gate                          ✅ Implemented

Query similarity                       ✅ Implemented

Popularity ranking                     ✅ Implemented

Streamlit application                  ✅ Implemented

Local launcher                         ✅ Implemented

Google Colab workflow                  ✅ Included

Optional lyrics integration            🟡 Ready; external data required

Optional audio integration             🟡 Ready; external data required

Human annotation workflow              🟡 Ready; annotations required

Human-validated song emotion metrics   ⏳ Not yet available



🧾 Engineering Decisions Worth Noticing



Why not force every song into an emotion?



Because a metadata-only system does not have enough evidence to make

that claim.



Why is emotion weighted 70%?



The application's primary purpose is emotion-conditioned recommendation,

so mood relevance should dominate query similarity and popularity.



Why keep low-confidence songs available?



Because metadata absence is not proof that a song is irrelevant. The

recommender therefore uses the complete six-way profile while giving

stronger evidence priority.



Why exclude artists from mood scoring?



Artist identity can leak historical associations into the recommendation

signal. The project keeps artist information for display while excluding

it from emotion/content scoring.



Why use a classical Linear SVM?



It is fast, lightweight, reproducible and highly suitable for sparse

TF-IDF text features. It also provides a strong baseline without

introducing unnecessary model complexity.



Why build a human-validation tool instead of claiming accuracy?



Because validation should come from real annotations rather than

assumptions. The repository makes that evaluation step reproducible

without fabricating evidence.



🛠️ Example Usage



After launching Moodify:



User:

"I had a really difficult day and feel lonely. I want

something emotional that helps me sit with the feeling."



The system:



1\. Vectorizes the text with TF-IDF

2\. Predicts one of six emotions with Linear SVM

3\. Converts SVM decision values into normalized ranking weights

4\. Uses the selected emotion to score catalogue relevance

5\. Computes TF-IDF similarity between the query and song metadata

6\. Adds a small popularity contribution

7\. Applies the evidence-first ranking logic

8\. Removes duplicate title/artist records

9\. Returns up to 10 recommendations



The user can then:



change the language



inspect the mood direction



shuffle the candidate set



open a track on Spotify



📁 Important Reports



The repository contains several engineering and validation documents:



File                                Purpose



emotion\_model\_report.md           Final supervised emotion-model

evaluation



emotion\_model\_metrics.json        Machine-readable evaluation metrics



FINAL\_FIX\_REPORT.md               Summary of corrective engineering

changes



final\_quality\_audit.md            Final quality and limitation audit



integration\_verification.md       Integration verification results



reports/LIMITATION\_STATUS.md      Evidence/limitation status



vad\_mapping\_experiment.md         VAD mapping experiment

documentation



LOCAL\_RUN\_GUIDE.md                Local execution guidance



DEPENDENCY\_MAP.md                 Project dependency mapping



🔐 Data \& Evidence Policy



Moodify intentionally follows a conservative rule:



If the repository does not contain evidence, the README does not

claim that the evidence exists.



Therefore:



Human song-emotion labels are not fabricated.



Lyrics are not fabricated.



Audio features are not fabricated.



Human-validation metrics are not fabricated.



Metadata relevance is explicitly described as heuristic.



SVM decision scores are not described as calibrated probabilities.



This makes the project easier to evaluate honestly and provides a clear

path for future research improvements.



👨‍💻 Author



Akshat Sajwan



B.Tech Computer Science student focused on:



Machine Learning



Natural Language Processing



Computer Vision



Python



Data Science



AI/ML application development



⭐ Final Takeaway



Moodify is a practical end-to-end ML project that combines NLP

classification, content-based recommendation, multilingual metadata

processing, ranking logic, Streamlit application development, validation

tooling, and responsible evidence handling.



Its strongest engineering characteristic is not simply the 89.05%

text-classification accuracy. It is the separation between:



What the model has actually learned

&#x20;               ↓

What the catalogue evidence actually supports

&#x20;               ↓

What still requires human or multimodal validation



That distinction keeps the current system technically defensible while

leaving a clear path toward a genuinely multimodal, human-validated

emotion-aware recommender.

