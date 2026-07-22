# Algright - Adaptive AI EEG Denoising & Classification Pipeline

> Brain-Computer Interface (BCI) research framework that uses Artificial Intelligence to clean noisy brain signals (EEG) and classify motor imagery tasks (left hand vs right hand movement intention).

---

## What Does This Project Do?

```
Noisy EEG Signal  -->  [Clean It]  -->  [Classify It]  -->  Brain-Computer Interface
```

1. **Generates or loads EEG data** (brain signals) - synthetic or real from PhysioNet dataset
2. **Cleans the noise** using 3 stages: traditional filters + adaptive filters + AI neural network
3. **Classifies** the cleaned signals using CSP features + SVM/LDA
4. **Self-learns** and adapts in real-time when the signal distribution changes
5. **Visualizes** everything with 13 publication-quality figures + a live web dashboard

---

## Project Structure

```
algright/
|
|-- config.py                   # All settings in one place (hyperparameters)
|-- run_pipeline.py             # Main script - run this to start everything
|-- run_streaming.py            # Stream EEG data over network (LSL)
|-- requirements.txt            # Full list of packages to install
|-- requirements_core.txt       # Minimal packages (if full list fails)
|
|-- adaptive_filter/            # Adaptive noise cancellation (LMS/RLS)
|-- ai_denoising/               # AI neural network for denoising (PyTorch DAE)
|-- classification/             # Feature extraction (CSP) + Classification (SVM/LDA)
|-- data_pipeline/              # Load datasets + stream EEG data (MOABB + LSL)
|-- evaluation/                 # Metrics (SNR, accuracy) + 13 publication figures
|-- preprocessing/              # Bandpass + Notch signal filters
|-- synthetic/                  # Generate fake EEG signals for testing
|-- self_learning/              # Online self-adaptation engine
|-- dashboard/                  # Live web dashboard (Flask + React)
|-- results/                    # Generated figure files (PNG)
```

---

## Step-by-Step Setup Guide (Follow Exactly)

### STEP 1: Install Python (if you don't have it)

You need **Python 3.9 or higher**. 

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click **"Download Python 3.x.x"** (latest version)
3. Run the installer
4. **IMPORTANT:** Check the box that says **"Add Python to PATH"** during installation
5. Click "Install Now"

**Verify it installed correctly.** Open Command Prompt (search "cmd" in Windows Start menu) and type:
```
python --version
```
You should see something like `Python 3.11.x`. If you see an error, Python is not added to PATH. Reinstall and check the box.

---

### STEP 2: Download This Project

**Option A - Using Git (Recommended):**
```
git clone https://github.com/Nitay2004/algright.git
cd algright
```

**Option B - Without Git:**
1. Go to [https://github.com/Nitay2004/algright](https://github.com/Nitay2004/algright)
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Extract the ZIP file
5. Open Command Prompt, navigate to the extracted folder:
```
cd C:\Users\YourName\Downloads\algright-main
```

---

### STEP 3: Create a Virtual Environment (Recommended but Optional)

A virtual environment keeps this project's packages separate from your other Python projects.

Open Command Prompt in the project folder and run:
```
python -m venv venv
```

Now activate it:
```
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your command prompt line. This means it's working.

> **Note:** Every time you open a new Command Prompt to work on this project, you must activate the venv again by running `venv\Scripts\activate` from the project folder.

---

### STEP 4: Install All Required Packages

**Option A - Full install (all features):**
```
pip install -r requirements.txt
```

**Option B - If full install fails, install core only:**
```
pip install -r requirements_core.txt
```

**What each package does:**

| Package | What It Does | Required? |
|---------|-------------|-----------|
| `numpy` | Math operations on arrays | YES |
| `scipy` | Signal filtering (bandpass, notch) | YES |
| `scikit-learn` | Machine learning (SVM, LDA, metrics) | YES |
| `torch` | AI neural network (denoising autoencoder) | YES |
| `mne` | EEG data handling + CSP feature extraction | YES |
| `matplotlib` | Generate plots and figures | YES |
| `pandas` | Data manipulation | YES |
| `padasip` | Adaptive filter algorithms (LMS/RLS) | Optional |
| `pylsl` | Real-time EEG streaming over network | Optional |
| `moabb` | Download PhysioNet EEG datasets | Optional |
| `braindecode` | Advanced EEG deep learning | Optional |
| `flask` | Web server for dashboard | Optional |
| `flask-socketio` | Real-time WebSocket communication | Optional |

**If `pip install` is slow, use:**
```
pip install -r requirements.txt -i https://pypi.org/simple/
```

**If you get permission errors:**
```
pip install -r requirements.txt --user
```

---

### STEP 5: Verify Installation

Run this command to check if everything is installed:
```
python -c "import numpy, scipy, sklearn, torch, mne, matplotlib; print('All packages installed successfully!')"
```

If you see `All packages installed successfully!`, you're good to go. If you see an error like `ModuleNotFoundError`, install that specific package:
```
pip install <package_name>
```

---

## How to Run the Project

### Run Mode 1: OFFLINE (Easiest - No internet needed)

This uses synthetic (fake) EEG data. Good for testing.

```
python run_pipeline.py --mode offline
```

**What happens:**
- Generates 200 fake EEG signals (64 channels, 250 Hz sampling rate)
- Adds noise to them
- Cleans them using 3 methods (filters + adaptive + AI)
- Classifies left vs right hand movement
- Generates 13 figure files in the `results/` folder
- Prints accuracy, SNR improvement, and latency to console

**Takes about 1-3 minutes depending on your computer.**

---

### Run Mode 2: DATASET (Needs internet, first time only)

This downloads real brain signal data from PhysioNet database.

```
python run_pipeline.py --mode dataset
```

**First run will download ~500MB of data** from PhysioNet. Subsequent runs use cached data.

**What happens:**
- Downloads PhysioNet Motor Imagery dataset (subjects 1, 2, 3)
- Processes real EEG signals through the full pipeline
- Same outputs as offline mode

---

### Run Mode 3: STREAMING (Needs 2 terminals)

This simulates real-time EEG streaming over network.

**Terminal 1** - Start the EEG data streamer:
```
python run_streaming.py
```

**Terminal 2** - Start the pipeline to receive and process:
```
python run_pipeline.py --mode streaming
```

---

### Run Mode 4: WEB DASHBOARD (Interactive UI)

```
python dashboard/backend.py
```

Then open your browser and go to: **http://localhost:5000**

You'll see a beautiful dark-themed dashboard with:
- Real-time EEG signal visualization
- Mode selection (Offline / Dataset / Live Stream)
- Parameter controls (channels, noise level, filter type)
- Live metrics (SNR, accuracy, latency)
- Training loss chart
- Confusion matrix
- ROC curve

---

## All Command-Line Options

```
python run_pipeline.py [OPTIONS]

Options:
  --mode {offline,dataset,streaming}   Execution mode (default: offline)
  --channels N                         Number of EEG channels (default: 64)
  --sfreq N                            Sampling frequency in Hz (default: 250)
  --noise-snr N                        Noise level in dB (default: -5.0)
                                        Higher = less noise, Lower = more noise
  --filter {lms,rls}                   Adaptive filter type (default: lms)
  --classifier {svm,lda}               Classifier type (default: svm)
```

**Examples:**
```
python run_pipeline.py --mode offline --channels 32 --sfreq 128
python run_pipeline.py --mode offline --noise-snr -10 --filter rls --classifier lda
python run_pipeline.py --mode dataset --channels 64
```

---

## Understanding the Output

### Console Output
After running, you'll see:
```
[Pipeline] Phase 1: Data generation...
[Pipeline] Phase 2: Preprocessing (bandpass + notch)...
[Pipeline] Phase 3: Adaptive filtering (LMS)...
[Pipeline] Phase 4: AI Denoising (Training DAE)...
[Pipeline] Phase 5: Classification (CSP + SVM)...
[Pipeline] Phase 6: Self-Learning Adaptation...
[Pipeline] Phase 7: Evaluation & Visualization...
```

### Generated Figures (in `results/` folder)

| File | What It Shows |
|------|--------------|
| `fig1_eeg_comparison.png` | Noisy vs Denoised vs Clean signal |
| `fig2_pipeline_stages.png` | All 5 stages of signal processing |
| `fig3_heatmap_Noisy_Input.png` | Heatmap of all channels (noisy) |
| `fig3_heatmap_Denoised_Signal.png` | Heatmap of all channels (denoised) |
| `fig4_frequency_spectrum.png` | Frequency power spectrum analysis |
| `fig5_dae_training.png` | AI model training loss curve |
| `fig6_snr_comparison.png` | Signal-to-noise ratio improvement |
| `fig7_confusion_matrix.png` | Classification accuracy matrix |
| `fig8_roc_curve.png` | ROC curve with AUC score |
| `fig9_latency_benchmark.png` | Processing speed benchmark |
| `fig10_latent_space.png` | t-SNE/PCA visualization |
| `fig11_adaptation_curve.png` | Self-learning adaptation over time |
| `fig12_architecture.png` | Neural network architecture diagram |
| `fig13_per_channel.png` | Per-channel denoising comparison |

---

## How the AI Pipeline Works (Technical Overview)

```
                    THE 6-STAGE PIPELINE
                    =====================

Stage 1: DATA ACQUISITION
    |-- Synthetic EEG (64 channels, 250 Hz)
    |-- OR PhysioNet Dataset via MOABB
    '-- OR Live LSL Stream

Stage 2: PREPROCESSING
    |-- Bandpass Filter (0.5 - 50 Hz) -- removes out-of-range frequencies
    '-- Notch Filter (50 Hz) -- removes powerline interference

Stage 3: ADAPTIVE FILTERING
    '-- LMS or RLS algorithm -- learns and cancels noise adaptively

Stage 4: AI DENOISING
    |-- Denoising Autoencoder (1D Convolutional Neural Network)
    |-- Encoder: 64 -> 32 -> 16 -> latent_dim (compressed representation)
    '-- Decoder: latent_dim -> 16 -> 32 -> 64 (reconstructed clean signal)

Stage 5: CLASSIFICATION
    |-- CSP (Common Spatial Pattern) -- extracts discriminative features
    '-- SVM or LDA -- classifies left vs right hand motor imagery

Stage 6: SELF-LEARNING
    |-- Detects distribution shift using KL-divergence
    |-- Triggers online adaptation when shift detected
    '-- Updates model in real-time without retraining from scratch
```

---

## Troubleshooting Common Issues

### "ModuleNotFoundError: No module named 'X'"
```
pip install X
```

### "torch.cuda.is_available() = False"
This is normal if you don't have an NVIDIA GPU. The project works on CPU. Training will be slower but functional.

### "pip install padasip fails"
Try:
```
pip install padasip --no-cache-dir
```
If it still fails, the adaptive filter module won't work but everything else will.

### "MOABB dataset download fails"
Make sure you have internet connection. If it keeps failing, use `--mode offline` instead.

### "Flask/SocketIO import error for dashboard"
```
pip install flask flask-socketio
```

### "Permission denied" during pip install
```
pip install -r requirements.txt --user
```

### "Python is not recognized as an internal or external command"
Reinstall Python and make sure to check **"Add Python to PATH"** during installation.

---

## Configuration

All settings are in `config.py`. You can change:

| Setting | Default | Description |
|---------|---------|-------------|
| `sfreq` | 250 | Sampling rate (Hz) |
| `n_channels` | 64 | Number of EEG channels |
| `l_freq` | 0.5 | Bandpass filter low cutoff (Hz) |
| `h_freq` | 50.0 | Bandpass filter high cutoff (Hz) |
| `notch_freq` | 50.0 | Notch filter frequency (Hz) |
| `filter_type` | "lms" | Adaptive filter: "lms" or "rls" |
| `lms_length` | 32 | LMS filter tap length |
| `lms_mu` | 0.01 | LMS learning rate |
| `latent_dim` | 32 | AI model latent space size |
| `batch_size` | 64 | Training batch size |
| `lr` | 1e-3 | Training learning rate |
| `epochs` | 30 | Training epochs |
| `csp_n_components` | 8 | CSP feature dimensions |
| `svm_kernel` | "rbf" | SVM kernel type |
| `noise_snr_db` | -5.0 | Input noise level (dB) |

---

## Requirements

- **Python** >= 3.9
- **OS** - Windows / macOS / Linux
- **RAM** - Minimum 4GB (8GB recommended)
- **GPU** - Optional (CUDA NVIDIA GPU speeds up AI training 5-10x)
- **Internet** - Required only for dataset download mode and dashboard

---

## License

This project is for research and educational purposes.

---

## Author

**Nitay** - [GitHub: Nitay2004](https://github.com/Nitay2004)
