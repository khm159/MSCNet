import os 
import shutil

def load_txt_from_path(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  return lines

def processing_labels(txt_file):
  label_dict = dict()
  for line in txt_file:
    split = line.split('\t')
    split = [x for x in split if x != '']
    if split[0]=='action_name':
      continue
    label_dict[split[0]] = split[1].split('\n')[0]
  return label_dict

def appending_data(txt_file, out_file, label_num):
  for line in txt_file:
    split = line.split('\n')[0]
    newline = split + ' ' + str(label_num) + '\n'
    out_file.write(newline)
    
def process_stanford40():
  label_txt = load_txt_from_path('./ImageSplits/actions.txt')
  label_dict = processing_labels(label_txt)
  actions = list(label_dict.keys())
  train_out = open('stanford40_train.txt', 'w')
  test_out = open('stanford40_test.txt', 'w')
  
  for lbl_num, act in enumerate(actions):
    # load train list 
    train_list = load_txt_from_path('./ImageSplits/'+act+'_train.txt')
    appending_data(train_list, train_out, lbl_num)
    # load test list 
    train_list = load_txt_from_path('./ImageSplits/'+act+'_test.txt')
    appending_data(train_list, test_out, lbl_num)

def process_stanford40_bm():
  label_txt = load_txt_from_path('./ImageSplits/actions.txt')
  label_dict = processing_labels(label_txt)
  actions = list(label_dict.keys())
  body_train_out = open('stanford40_body_train.txt', 'w')
  body_test_out = open('stanford40_body_test.txt', 'w')
  non_body_train_out = open('stanford40_non_body_train.txt', 'w')
  non_body_test_out = open('stanford40_non_body_test.txt', 'w')
  body_actions = ['climbing', 'jumping','cleaning_the_floor','riding_a_bike','riding_a_horse',
                  'rowing_a_boat','running','walking_the_doc','shooting_an_arrow','throwing_frisby',
                  'waving_hands']
  for lbl_num, act in enumerate(actions):
    # load train list 
    train_list = load_txt_from_path('./ImageSplits/'+act+'_train.txt')
    # load test list 
    test_list = load_txt_from_path('./ImageSplits/'+act+'_test.txt')

    if act in body_actions:
      # body motion labels
      appending_data(train_list, body_train_out, lbl_num)
      appending_data(test_list, body_test_out, lbl_num)
    else:
      # non body-motion labels
      appending_data(train_list, non_body_train_out, lbl_num)
      appending_data(test_list, non_body_test_out, lbl_num)
    
if __name__ =="__main__":
    process_stanford40()
    process_stanford40_bm()
