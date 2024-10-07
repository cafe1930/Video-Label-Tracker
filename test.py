import pandas as pd
import os
import glob
from tqdm import tqdm



if __name__ == '__main__':
    # Первая итерация новые рамки, полученные от "нейронки"
    paths_to_txts_list = glob.glob(r'i:\AVABOS\!!! ТРЭКИНГ НА РАЗМЕТКУ\!!! DONE\*\*.txt')

    
    for path in tqdm(paths_to_txts_list):
    #for path in paths_to_txts_list:
        path_to_root, txt_name = os.path.split(path)
        csv_name = txt_name.split('_persons_descr.txt')[0]
        csv_name = f'{csv_name}.csv'
        path_to_csv = os.path.join(path_to_root, csv_name)
        
        with open(path, encoding='utf-8') as fd:
            obj_descr_list = [descr for descr in fd.read().split('\n') if descr != '']

        df = pd.DataFrame(columns=['object_idx', 'class_name', 'object_description'])
        for idx, obj_descr in enumerate(obj_descr_list):
            df.loc[idx] = {'object_idx': idx, 'class_name': 'person', 'object_description': obj_descr.strip()}
        df.to_csv(path_to_csv, index=False)
        #df_str = txt_name+'\n'+str(df) + '\n----------------------------------\n'
        #with open('log.txt', 'a') as fd:
        #    fd.write(df_str)
        #print(df)
        #print('----------------------------------')
