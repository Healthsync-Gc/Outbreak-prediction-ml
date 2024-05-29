## Installation Instructions

### 1. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Set Up the Virtual Environment (Optional but Recommended)

Create a virtual environment to manage dependencies:

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist, create it with the following content:

```
pandas
prophet
```

Then run the command above.

### 4. Prepare Your Data

Ensure your data files are in the `data` directory. Each file should be named according to the region (e.g., `addis_ababa.csv`, `amhara.csv`, etc.) and contain the following columns:

- `date`: The date column
- `value`: The value column to be forecasted

### 5. Run the Training Script

Execute the training script to train and save the models:

```bash
python train.py
```

### 6. Verify the Models

After running the script, the trained models will be saved in the `models` directory. Each model will be saved as a pickle file named after the corresponding region (e.g., `addis_ababa_model.pkl`, `amhara_model.pkl`, etc.).