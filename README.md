# Hiver Support Agent

An AI-assisted customer support system that analyzes customer messages, identifies their intent, retrieves similar support cases, generates a grounded response, and decides whether the request should be handled automatically or escalated.

## Features

* Customer intent classification using TF-IDF and Logistic Regression
* Retrieval of similar historical support conversations
* Brand-aware case retrieval
* Confidence and risk-based escalation
* Grounded response generation
* Evaluation of classification and escalation performance
* Failure analysis for incorrect or unsafe responses
* Reproducible evaluation using a fixed random seed

## Project Workflow

```text
Customer Message
       ↓
Text Preprocessing
       ↓
Intent Classification
       ↓
Confidence & Risk Analysis
       ↓
Historical Case Retrieval
       ↓
Response Generation
       ↓
Safety Check
       ↓
AUTO_HANDLE / ESCALATE
```

## Dataset

The project uses the **Customer Support on Twitter** dataset.

For practical development and evaluation, the project uses a smaller working set of five brands:

* AmazonHelp
* AppleSupport
* Uber_Support
* SpotifyCares
* Delta

The data is split at the conversation/thread level to reduce train-test leakage.

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* TF-IDF
* Logistic Regression
* Matplotlib
* YAML configuration
* Optional Anthropic API integration

## Installation

Create and activate a virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```cmd
pip install -r requirements.txt
```

## Run the Project

Run the pipeline for a brand:

```cmd
python run_pipeline.py --brand AppleSupport --sample
```

Run the demo:

```cmd
python demo.py
```

Run evaluation:

```cmd
python evaluate.py --golden_set data/evaluation/golden_set.csv
```

Run the test suite:

```cmd
python -m unittest discover tests
```

## Evaluation

The project evaluates:

* Intent classification performance
* Comparison against a majority-class baseline
* Retrieval behaviour
* Escalation decisions
* Failure cases involving incorrect automatic handling

The escalation policy is designed to be conservative because automatically handling an uncertain or sensitive customer request can be more problematic than escalating it.

During development, the escalation policy was adjusted to treat uncertain `other` predictions more conservatively. This reduced incorrect automatic handling while increasing the number of escalations.

More details are available in `FAILURE_ANALYSIS.md`.

## Evaluation Set

The project includes a 240-example evaluation set selected from the test data.

The examples were reviewed and labelled with AI assistance. The evaluation set uses a single-annotator process and does not provide independent human annotation or inter-annotator agreement.

This limitation is documented so that the evaluation results are not presented as independently human-validated ground truth.

## Current Limitations

* Training labels were generated using keyword/rule-based weak labelling.
* The evaluation set does not have independent human annotation.
* Some intents contain relatively few evaluation examples.
* The default response generator is template-based.
* The optional LLM evaluation component requires an API key.
* The escalation policy is conservative and may produce additional escalations.

## Project Structure

```text
hiver-support-agent/
│
├── config/
├── data/
├── docs/
├── outputs/
├── scripts/
├── src/
├── tests/
│
├── demo.py
├── evaluate.py
├── run_pipeline.py
├── requirements.txt
├── REPORT.md
└── README.md
```

## Main Components

### Classification

Predicts the intent of an incoming customer support message using TF-IDF features and Logistic Regression.

### Retrieval

Finds relevant historical support conversations using text similarity and brand information.

### Escalation

Uses prediction confidence and risk-related rules to decide whether a request should be automatically handled or escalated.

### Generation

Produces a response grounded in the retrieved support information.

### Evaluation

Measures model performance and analyses cases where the system makes incorrect or risky decisions.

## Reproducibility

The project uses fixed random seeds where applicable and stores processed datasets and evaluation outputs.

Additional technical information is available in:

* `REPORT.md`
* `FAILURE_ANALYSIS.md`
* `DECISION_LOG.md`
* `docs/INTENT_TAXONOMY.md`

## License

MIT
