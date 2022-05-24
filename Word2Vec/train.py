import os
import time
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

import corpus


def printTime():
    t = time.localtime()
    print('[{h}:{m}:{s}] '.format(h=t.tm_hour, m=t.tm_min, s=t.tm_sec), end='')

#%%
# https://datascience.stackexchange.com/questions/9819/number-of-epochs-in-gensim-word2vec-implementation
class LossLogger(CallbackAny2Vec):
    # Output loss at each epoch
    def __init__(self):
      self.epoch = 1
      self.losses = []

    def on_epoch_begin(self, model):
      printTime()
      print('\nEpoch: {epoch}'.format(epoch=self.epoch))

    def on_epoch_end(self, model):
      loss = model.get_latest_training_loss()
      self.losses.append(loss)
      print('  Loss: {loss}'.format(loss=loss))
      self.epoch += 1                       

#%%
# main function
if __name__ == '__main__':
    wiki_dump_path = '../dataset/zhwiki.xml.bz2'
    healthdoc_path = '../dataset/HealthDoc/'
    
    wiki_pkl_path = '../dataset/zhwiki.pkl'
    healthdoc_pkl_path = '../dataset/healthdoc.pkl'

    model_output_path = '../model/word2vec-healthdoc-wiki-300.model'

    # model parameter
    vector_size = 300;
    workers = 2
    epochs = 5
    min_count = 10 # vocabulary under min_count would be ignore
    loss_logger = LossLogger()

    # download latest wiki dump
    printTime()
    corpus.download_wiki_dump(wiki_dump_path)

    # build pkl dataset for HealthDoc and zh-wiki dump
    printTime()
    corpus.createPKL_healthdoc(healthdoc_path)
    printTime()
    corpus.createPKL_wiki(wiki_dump_path)

    # parse corpus
    printTime()
    print("Parsing corpus")
    sentences = corpus.CorpusSentencesPKL(wiki_pkl_path, healthdoc_pkl_path)

    printTime()
    print('Training Word2Vec model')
    model = Word2Vec(sentences, 
                     sg=1, 
                     vector_size=vector_size, 
                     workers=workers, 
                     epochs=epochs, 
                     min_count=min_count, 
                     callbacks=[loss_logger], 
                     compute_loss=True)

    printTime()
    print('Training done.')

    printTime()
    print('Save trained word2vec model to "{path}"'.format(path=model_output_path))
    model.save(model_output_path)
    #
    #with open(model_output_path, 'w', encoding='utf-8') as f:
    #  f.write('%d %d\n' % (len(model.wv.vocab), vector_size))
    #  for word in tqdm(model.wv.vocab):
    #    f.write('%s %s\n' % (word, ' '.join([str(v) for v in model.wv[word]])))
    #
    printTime()
    print('Done')
