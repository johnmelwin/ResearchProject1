
# Project Data Overview 📊

This folder contains datasets crucial for training and validating models in our software engineering project. The datasets are organized into specific categories for different phases of model training and verification.

## Data Organization 🗂️

### 1. Data for Initial Pre-Training (Sample) 🛠️
- **Folder:** `Data_First_PreTraining(Sample)`
- **Contents:**
  - 📄 `Code_Instruction_Tuning.json`
    - Description: Contains code data formatted for instruction tuning.
  - 📄 `Traceability_random_methods.json`
    - Description: A sample dataset used to train all models, focusing on traceability.

### 2. Data for Model Verification 📋
- **Folder:** `Data_Model_Verifcation`
- **Contents:**
  - 📄 `databricks-dolly-15k.jsonl`
    - Description: Dataset used for testing the Llama-2-hf model.

### 3. Data for Second Pre-Training (Unformatted) 🧩
- **Folder:** `Data_Second_PreTraining(unformated)`
- **Contents:**
  - 🗃️ Contains data from four projects with traces between methods, requirements, and classes.
  - Note: This data needs to be formatted into the instruction tuning format.

---

This README provides an overview of the data structure and purpose within the project, ensuring efficient navigation and understanding of the datasets used.
