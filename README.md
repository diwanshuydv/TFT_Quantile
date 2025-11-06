conda create -n tft_trading python=3.9

conda activate tft_trading

conda install -c rapidsai -c conda-forge -c nvidia cudf=22.02 dask-cudf=22.02 python=3.9 cudatoolkit=11.5

python -c "import cudf; print(cudf.__version__)"

pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

python main_enhanced.py

# Skip already completed steps
python main_enhanced.py --skip-conversion --skip-data-prep

# Adjust hyperparameter tuning
python main_enhanced.py --tuning-trials 100

# Customize data splits
python main_enhanced.py --train-end 200 --val-end 240

# Skip tuning (use default hyperparameters)
python main_enhanced.py --skip-tuning

# Only backtest (requires trained model)
python main_enhanced.py --skip-conversion --skip-data-prep --skip-tuning --skip-training

My architecture is:-
Input (batch, 120, 471) 
Variable Selection Network
LSTM (3 layers, hidden=128)
Temporal Fusion Decoder
Multi-Head Attention (8 heads) × 2
Position-wise FFN (256 hidden)
Gated Output Layer
Classification (3 classes: Short/Neutral/Long)

Total Parameters: ~500,000
