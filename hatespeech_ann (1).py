#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Kurulması gereken paketler
get_ipython().system('pip install tensorflow scipy==1.10.1 keras-tuner --quiet')


# In[2]:


get_ipython().system('pip install shap')


# In[3]:


get_ipython().system('pip install numpy --upgrade')


# In[1]:


get_ipython().system('pip install numpy==1.21.6 scipy==1.9.3 shap --user --force-reinstall --no-cache-dir')


# In[5]:


pip install datasets


# In[6]:


pip install transformers datasets sentencepiece


# In[7]:


get_ipython().system('pip uninstall -y keras')
get_ipython().system('pip install keras==2.12.0')
get_ipython().system('pip install tf-keras')


# In[8]:


get_ipython().system('pip uninstall -y torch')


# In[9]:


pip install tqdm


# In[10]:


get_ipython().system('pip uninstall -y transformers accelerate')
get_ipython().system('pip install transformers==4.38.2 accelerate==0.25.0')


# In[11]:


get_ipython().system('pip install --upgrade torch==2.3.0+cpu -f https://download.pytorch.org/whl/torch_stable.html')


# In[12]:


get_ipython().system('pip uninstall -y numpy pandas')
get_ipython().system('pip install numpy==1.24.4 pandas==1.5.3')


# In[3]:


get_ipython().system('pip uninstall -y numba llvmlite shap')
get_ipython().system('pip install numba==0.56.4 llvmlite==0.39.1 shap')


# In[2]:


get_ipython().system('pip uninstall -y backports')


# In[2]:


get_ipython().system('pip uninstall -y shap numba numpy pandas')


# In[3]:


get_ipython().system('pip install shap')


# In[2]:


get_ipython().system('pip install --upgrade --force-reinstall numpy==1.21.6 scipy==1.9.3 --user')


# In[1]:


import sys
sys.modules.pop('scipy', None)
sys.modules.pop('scipy.linalg', None)
sys.modules.pop('scipy.linalg.special_matrices', None)


# In[35]:


#Kütüphaneler
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import FastText
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import torch
import logging
from datetime import datetime
from tensorflow.keras.models import load_model
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import torch
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from transformers import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import DatasetDict
from contextlib import nullcontext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
from datasets import Dataset
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional, Input, Concatenate, Attention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tempfile
import os
import shutil
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Attention, GlobalAveragePooling1D, Concatenate, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import tensorflow as tf


# In[3]:


#Stopwords çıkarma
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[4]:


#Veri okuma
data = pd.read_csv("HateSpeechDatasetBalanced.csv", names=["Content", "Label"], skiprows=1)


# In[5]:


# Eksik değer kontrolü
data.dropna(subset=["Content", "Label"], inplace=True)
data = data[data["Content"].str.strip() != ""] 


# In[6]:


data.head()


# In[7]:


#Veriyi  temizleme
def is_clean_text(text):
    text = str(text).strip()
    if len(text.split()) < 3:
        return False
    if re.fullmatch(r'[\W_]+', text):
        return False
    return True

hate_words = ["slut", "fuck", "jew", "kill", "nazi", "pig", "faggot", "die", "bitch", "cunt", "retard"]

def possible_hate(text):
    return any(word in text.lower() for word in hate_words)

filtered_data = data[data['Content'].apply(is_clean_text)]
filtered_data = filtered_data[~((filtered_data['Label'] == 0) & (filtered_data['Content'].apply(possible_hate)))]


# In[8]:


#Etkiketlenmiş veri setini dengeleme
min_count = filtered_data['Label'].value_counts().min()
balanced_data = pd.concat([
    filtered_data[filtered_data['Label'] == 0].sample(min_count, random_state=42),
    filtered_data[filtered_data['Label'] == 1].sample(min_count, random_state=42)
]).sample(frac=1, random_state=42)


# In[9]:


#Data agumentation
def augment_text(text):
    words = text.split()
    if len(words) > 3:
        idx = random.randint(0, len(words) - 1)
        words[idx] = "<unk>"  # Bilinmeyen kelime simülasyonu
    return " ".join(words)


augmented_texts = balanced_data['Content'].apply(augment_text)
augmented_labels = balanced_data['Label']


augmented_df = pd.DataFrame({'Content': augmented_texts, 'Label': augmented_labels})
balanced_data = pd.concat([balanced_data, augmented_df]).sample(frac=1, random_state=42)


# In[10]:


#Tokenizer kısmı 
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(balanced_data['Content'])
sequences = tokenizer.texts_to_sequences(balanced_data['Content'])
max_len = 100
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
labels = balanced_data['Label'].values


# In[11]:


#Cümle uzunluğu bulma ve normalizasyon
lengths = balanced_data['Content'].apply(lambda x: len(x.split())).values.reshape(-1, 1)

scaler = MinMaxScaler()
normalized_lengths = scaler.fit_transform(lengths)


# In[12]:


#Fasttext embedding
tokenized_sentences = [text.lower().split() for text in balanced_data['Content']]
ft_model = FastText(sentences=tokenized_sentences, vector_size=300, window=5, min_count=3, workers=4, sg=1, epochs=10)


# In[13]:


# Train/Test
X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
    padded_sequences, labels, normalized_lengths, test_size=0.2, random_state=42)

# Train/Validation
X_train, X_val, y_train, y_val, len_train, len_val = train_test_split(
    X_train, y_train, len_train, test_size=0.2, random_state=42)


# In[14]:


#Embedding matrisi
embedding_matrix = np.zeros((10000, 300))
for word, i in tokenizer.word_index.items():
    if i < 10000 and word in ft_model.wv:
        embedding_matrix[i] = ft_model.wv[word]


# In[52]:


#Attention katmanı

import tensorflow as tf

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name='att_weight',
            shape=(feature_dim, 1),
            initializer='random_normal',
            trainable=True,
            dtype=tf.float32
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(1,),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return output 
         


# In[53]:


#Early stopping overfitting oluşmasın diye
early_stop = EarlyStopping(
    monitor='val_loss',        
    patience=2,                  
    restore_best_weights=True   
)


# In[54]:


#Keras Tuner ile hiperparametre araması
def build_model(hp):
    input_seq = Input(shape=(max_len,))
    input_len = Input(shape=(1,))

 
    embedding_layer = Embedding(
        input_dim=10000,
        output_dim=300,  # SABİT
        weights=[embedding_matrix],
        trainable=True
    )(input_seq)

    lstm_units = hp.Int("lstm_units", min_value=32, max_value=128, step=32)
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding_layer)

    attention_out = Attention()(bilstm)

    pooled = GlobalAveragePooling1D()(attention_out)

    concat = Concatenate()([pooled, input_len])

    dropout_rate = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)
    dropout = Dropout(dropout_rate)(concat)

    output = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=[input_seq, input_len], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# In[55]:


#"tuner_dir" dosyası için kontrol
path = "tuner_dir"

if os.path.exists(path):
    if os.path.isfile(path):
        os.remove(path)  
        print("Dosya silindi: tuner_dir")
    elif os.path.isdir(path):
        shutil.rmtree(path)  
        print("Klasör silindi: tuner_dir")
else:
    print("Zaten yok: tuner_dir")


# In[56]:


#RandomSearch için tuner nesnesi
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=1,
    directory=tempfile.gettempdir(),
    project_name="lstm_tuning",
    overwrite=True
)


# In[57]:


#Search kısmı
tuner.search([X_train, len_train], y_train,
             validation_data=([X_val, len_val], y_val),
             epochs=3,
             batch_size=32)


# In[58]:


#En iyi hiperparametreleri kaydetme 
best_hp = tuner.get_best_hyperparameters(1)[0]

best_lstm_units = best_hp.get("lstm_units")
best_dropout = best_hp.get("dropout")
print(f"Best LSTM units: {best_lstm_units}, Best Dropout: {best_dropout}")


# In[59]:


# Öğrenme oranını iyileştirme
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',  
    factor=0.5,              
    patience=2,              
    min_lr=1e-6,             
    verbose=1
)


# In[60]:


#Model checkpoint
checkpoint = ModelCheckpoint(
    filepath='best_lstm_model.h5',     
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)


# In[63]:


#BiLSTM cümle sınıflandırma modeli ve model summary
input_seq = Input(shape=(max_len,))     
input_len = Input(shape=(1,))         


embedding_layer = Embedding(input_dim=10000, output_dim=300,
                            weights=[embedding_matrix],
                            trainable=True)(input_seq)

bilstm = Bidirectional(LSTM(96, return_sequences=True,
                            kernel_regularizer=l2(0.001)))(embedding_layer)


attention_out = Attention()(bilstm) 


pooled = GlobalAveragePooling1D()(attention_out)


concat = Concatenate()([pooled, input_len])

dropout = Dropout(0.2)(concat)
output = Dense(1, activation='sigmoid')(dropout)


model = Model(inputs=[input_seq, input_len], outputs=output)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit([X_train, len_train], y_train,
          validation_data=([X_val, len_val], y_val),
          epochs=10,
          batch_size=32,
          callbacks=[early_stop,reduce_lr,checkpoint])

model.summary()


# In[64]:


#Modeli kaydetme
model.save("lstm_model.h5")


tokenizer_json = tokenizer.to_json()
with open("tokenizer_lstm.json", "w") as f:
    f.write(tokenizer_json)


# In[65]:


#Train/validation accuracy-loss grafikleri
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# In[67]:


# Modeli ve tokenizeri yükleme
model_lstm = load_model("lstm_model.h5", custom_objects={'Attention': Attention})

with open("tokenizer_lstm.json") as f:
    tokenizer_json = f.read()  
tokenizer_lstm = tokenizer_from_json(tokenizer_json)


max_len = 100 


# In[68]:


def lstm_predict(texts):
    seqs = tokenizer_lstm.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=max_len, padding='post')
    lengths = np.array([[len(t.split()) / max_len] for t in texts])
    return model_lstm.predict([padded, lengths])


# In[71]:


get_ipython().system('pip install shap')


# In[72]:


#Shap ile model tahminine açıklık getirme
import shap
explainer = shap.Explainer(lstm_predict, shap.maskers.Text(" "))
shap_values = explainer(["You are pathetic and disgusting."])
shap.plots.text(shap_values[0])


# In[73]:


#Test verisi ile model performansı
model.evaluate([X_test, len_test], y_test)


# In[74]:


# s-nlp/ParaDetox
para1 = load_dataset("s-nlp/ParaDetox", split="train")

#  textdetox/multilingual_paradetox - ingilizce kısmı
para2 = load_dataset("textdetox/multilingual_paradetox", split="en")
print("Para1 örneği:", para1[0])
print("Para2 örneği:", para2[0])


# In[75]:


# Farklı kaynaklardan verileri birleştirip HuggingFace Dataset formatına çevirme
combined_data = []


for item in para1:
    combined_data.append({
        "input_text": item["en_toxic_comment"],
        "target_text": item["en_neutral_comment"]
    })


for item in para2:
    combined_data.append({
        "input_text": item["toxic_sentence"],
        "target_text": item["neutral_sentence"]
    })

combined_dataset = Dataset.from_list(combined_data)
print(combined_dataset[0])
print(f"Toplam örnek sayısı: {len(combined_dataset)}")


# In[76]:


# Birleşik veriyi CSV dosyasına kaydetme

df_combined = pd.DataFrame(combined_data)
df_combined.to_csv("combined_paradetox.csv", index=False)

print(" combined_paradetox.csv dosyası başarıyla kaydedildi.")


# In[77]:


# CSV verisini yükleyip train/validation olarak ayırma
dataset = load_dataset("csv", data_files="combined_paradetox.csv")


train_dataset = dataset["train"]


split_dataset = train_dataset.train_test_split(test_size=0.1)


print("Train örnek:", split_dataset["train"][0])
print("Validation örnek:", split_dataset["test"][0])


# In[78]:


# T5 modelini ve tokenizer'ını yükleme (google-t5/t5-small)
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")


# In[79]:


# T5 için giriş ve hedef metinleri tokenize edip modele uygun formata çevirme

def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    
    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=128, truncation=True, padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]

 
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["labels"]
    }



# In[80]:


tokenized_dataset = dataset.map(preprocess_function, remove_columns=dataset["train"].column_names)


# In[81]:


# Tokenize edilmiş veriyi ve ilk örneği görüntüleme

print(tokenized_dataset)
print(tokenized_dataset["train"][0])


# In[82]:


# T5 model eğitimi için eğitim ayarlarını tanımlama

training_args = TrainingArguments(
    output_dir="./t5-detoxified-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100
)


# In[84]:


# Eğitim verisi için dataLoader oluşturma

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,              
    return_tensors="pt",      
)

train_dataloader = DataLoader(
    tokenized_dataset["train"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator
)


# In[85]:


# İlk batch'in tensör boyutlarını görüntüleme

for batch in train_dataloader:
    print({k: v.shape for k, v in batch.items()})
    break


# In[86]:


# Önceden eğitilmiş T5 modelini veriye özel olarak fine-tune etme döngüsü

optimizer = AdamW(model.parameters(), lr=5e-5)


for epoch in range(10):
    total_loss = 0
    loop = tqdm(train_dataloader, leave=True)
    for batch in loop:
        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())
    
    print(f"Epoch {epoch+1}: Total Loss = {total_loss:.4f}")


# In[87]:


#Modeli kaydetme
model.save_pretrained("t5-detoxified")
tokenizer.save_pretrained("t5-detoxified")


print("Model başarıyla kaydedildi.")


# In[88]:


# T5 encoder içindeki feed-forward katmanın aktivasyonlarını görselleştirme

model = T5ForConditionalGeneration.from_pretrained("t5-detoxified")
tokenizer = T5Tokenizer.from_pretrained("t5-detoxified")
model.eval()


activations = []


def hook_fn(module, input, output):
    activations.append(output.detach().cpu())


target_layer = model.encoder.block[0].layer[1]  
hook = target_layer.register_forward_hook(hook_fn)


text = "You are so stupid and ugly"
inputs = tokenizer(text, return_tensors="pt")


with torch.no_grad():
    _ = model.encoder(**inputs)


hook.remove()

act = activations[0].squeeze().numpy()  

plt.figure(figsize=(10, 6))
plt.hist(act.flatten(), bins=50, color='green')
plt.title("Transformer Katmanı Aktivasyon Histogramı (FFN)")
plt.xlabel("Activation Değeri")
plt.ylabel("Frekans")
plt.show()


# In[89]:


# Kayıtlı tokenizer'ı JSON dosyasından yükleme

with open("tokenizer_lstm.json", "r", encoding="utf-8") as f:
    tokenizer_json = f.read()

tokenizer_lstm = tokenizer_from_json(tokenizer_json)


# In[90]:


# Fine-tune edilmiş T5 modelini ve tokenizer'ını yükleme

model_transformer = T5ForConditionalGeneration.from_pretrained("t5-detoxified")
tokenizer_transformer = T5Tokenizer.from_pretrained("t5-detoxified")


# In[92]:


# Eğitilmiş LSTM modelini dosyadan yükleme

model_lstm = load_model("lstm_model.h5", custom_objects={"Attention": Attention})


# In[113]:


# Kullanıcıdan cümle alıp LSTM ile tahmin yapma, gerekiyorsa transformers modeli ile düzeltme

def predict_input():
    while True:
        text = input("\nCümle gir (çıkmak için q): ")
        if text.lower() == 'q':
            break

      
        seq = tokenizer_lstm.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        length_input = np.array([[len(text.split()) / max_len]])

        pred = model_lstm.predict([padded, length_input])[0][0]

        if pred >= 0.5:
            print(f"Prediction: {pred:.4f} → Hate Speech")

          
            inputs = tokenizer_transformer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model_transformer.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            corrected = tokenizer_transformer.decode(outputs[0], skip_special_tokens=True)
            print(f"→ Rewriting Sentence: {corrected}")

        else:
            print(f"Prediction: {pred:.4f} → Not Hate")
            print(f"→ Sentence: {text}")


# In[114]:


# Tahmin başlatma

predict_input()


# In[115]:


# Web arayüzü üzerinden hate speech tahmini ve düzeltme servisi

app = Flask(__name__)


logging.basicConfig(filename='model_monitoring.log', level=logging.INFO)

def log_prediction(input_text, prediction_score, label):
    logging.info(f"{datetime.now()} | Input: {input_text} | Score: {prediction_score:.4f} | Label: {label}")


HTML_FORM = """
<!DOCTYPE html>
<html>
<head><title>Hate Speech Prediction</title></head>
<body>
    <h2>Enter a sentence:</h2>
    <form method="post" action="/predict">
        <input type="text" name="text" style="width:400px;">
        <input type="submit" value="Analyze">
    </form>
    {% if prediction %}
        <h3>Prediction: {{ prediction }}</h3>
        <p><strong>Result:</strong> {{ result_text }}</p>
    {% endif %}
</body>
</html>
"""


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    result_text = None

    if request.method == 'POST':
        text = request.form['text']

        # LSTM tahmini
        seq = tokenizer_lstm.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        length_input = np.array([[len(text.split()) / max_len]])
        pred = model_lstm.predict([padded, length_input])[0][0]

        if pred >= 0.5:
            prediction = f"{pred:.4f} → Hate Speech"

         
            inputs = tokenizer_transformer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model_transformer.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            corrected = tokenizer_transformer.decode(outputs[0], skip_special_tokens=True)
            result_text = f"Rewriting sentence: {corrected}"
           

          
            log_prediction(text, pred, "Hate Speech")

        else:
            prediction = f"{pred:.4f} → Not Hate"
            result_text = f"Input sentence: {text}"

           
            log_prediction(text, pred, "Not Hate")

    return render_template_string(HTML_FORM, prediction=prediction, result_text=result_text)


if __name__ == "__main__":
    from threading import Thread
    Thread(target=app.run, kwargs={'debug': True, 'use_reloader': False}).start()


# In[96]:


# LSTM modelinin test sonuçları için confusion matrix ve classification report

y_pred = model_lstm.predict([X_test, len_test])
y_pred_classes = (y_pred >= 0.5).astype(int)


cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LSTM Model Confusion Matrix")
plt.show()


print(classification_report(y_test, y_pred_classes))


# In[97]:


# ROUGE metriği ile metin tahmin performansını değerlendirme
rouge = evaluate.load("rouge")


predicted_texts = [
    "You are a useless person.",
    "Go kill yourself, nobody care about you.",
    "That group should be wiped out"
]

reference_texts = [
    "You are such a stupid and useless person!",
    "Go kill yourself, nobody cares about you.",
    "That group should be wiped off the planet"
]


results = rouge.compute(predictions=predicted_texts, references=reference_texts)
print(results)


# In[ ]:




