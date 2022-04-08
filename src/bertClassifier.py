import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import *
import os

def bert_tokenizer(tokenizer, sent, MAX_LEN):
    encoded_dict=tokenizer.encode_plus(
    text = sent,
    add_special_tokens=True,
    max_length=MAX_LEN,
    pad_to_max_length=True,
    return_attention_mask=True,
    truncation = True)

    input_id=encoded_dict['input_ids']
    attention_mask=encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']

    return input_id, attention_mask, token_type_id


class TFBertClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super(TFBertClassifier, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_class,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range),
                                                name="classifier")

    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        #outputs 값: sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)

        return logits


def train_model(model, claims, labels, epoch_num, batch_size, patience, min_delta):
    # preparing for training
    optimizer = tf.keras.optimizers.Adam(3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    model_name = "tf_bert_classifier"

    # earlystop to prevent overfitting
    earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=min_delta, patience=patience)
    # min_delta: the threshold that triggers the termination (acc should at least improve 0.0001)
    # patience: stop if 5 no improvment epochs

    checkpoint_path = os.path.join(model_name, 'weights.h5')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create path if exists
    if os.path.exists(checkpoint_dir):
        print("{} -- Folder already exists \n".format(checkpoint_dir))
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print("{} -- Folder create complete \n".format(checkpoint_dir))

    cp_callback = ModelCheckpoint(
        checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

    # train and evaluate
    history = model.fit(claims, labels, epochs=epoch_num, batch_size=batch_size,
                            validation_split=0.15, callbacks=[earlystop_callback, cp_callback])
    return model, history