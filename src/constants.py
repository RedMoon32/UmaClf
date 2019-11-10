# ============ Data Preparation ===========
zip_f = 'image_zip'
target_path = 'task'
url = 'https://cloud.uma.tech/index.php/s/256FPfW6pDypr62/download'
csv_inside_path = 'images_labelling.csv'
token = '' # put your token
# ============ Model Training ==========
seed = 42
batch_size = 64
test_batch_size = 100
epochs = 10
test_split = 0.2
log_interval = 700
lr = 0.005
out_model_path = "footballers.pt"

# ============ Bot Messages ==============
greeting = "Send me footballer⚽️ picture by photo and i will classify it!"
result_clf = "⚽ ️Image Class: {}"
