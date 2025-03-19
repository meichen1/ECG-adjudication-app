
import sys, os
import pandas as pd
import numpy as np
sys.path.append('/workspace/DataPreproc/')
from ECGXMLReader250221 import ECGXMLReader, ecg_plot_4leads, ecg_plot12   
batch_num = int(os.getenv('BATCH')) if os.getenv('BATCH') else ValueError('BATCH number not provided')
num_plots_perbatch = 50


def fun_image_whole_folder(UncertainMasterlistcsv, batch, img_dir, num_plots_perbatch = num_plots_perbatch, plot_12leads = False):
    
    # # Recursively find XML files
    # xml_files = [y for x in os.walk(input_folder) for y in glob.glob(os.path.join(x[0], '*.xml'))]
    # xml_files.sort()
    # pd.DataFrame(xml_files, columns=['xml_file']).to_csv('/workspace/ECGXML/uncertain_files_fullpath.csv', index=False, header=False)
    
    processed = 0
    startIdx = (batch-1)*num_plots_perbatch
    xml_files = pd.read_csv(UncertainMasterlistcsv, header=None).iloc[startIdx:startIdx+num_plots_perbatch,0].values
    processed_files = []
    
    for xml_file in xml_files:
        
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        
        ecg = ECGXMLReader(xml_file, augmentLeads=True)
        patient_id = ecg.patientId
        if len(patient_id) != 8:
            patient_id = '*' * (8 - len(patient_id)) + patient_id
        acquisition_date = ecg.ECGtime
        # diagnosis = ecg.diagnosis
        
        specified_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        all_voltages = ecg.getAllVoltages()
        ordered_voltages = {lead: all_voltages[lead] for lead in specified_order if lead in all_voltages}
        df = np.array(list(ordered_voltages.values())).astype('float32')

        ecg_plot_4leads(df, title = 'ECG 4 - Patient ID: ' +patient_id +'; Date: '+ str(acquisition_date), diagnosis_note= None, save_fig=True, save_fig_name=img_dir+patient_id+'_'+str(acquisition_date).replace(' ', '_')+'_04leads.png', show_fig=False)
        
        if plot_12leads:
            ecg_plot12(df[0:6, :], sample_rate = 250, title = 'ECG 12a - ID: ' +patient_id , save_fig=True, save_fig_name= img_dir+patient_id+'_'+str(acquisition_date).replace(' ', '_')+'_12leads_a.png')
            ecg_plot12(df[6:12, :], sample_rate = 250, title = 'ECG 12b - ID: ' +patient_id + patient_id[-4:], save_fig=True, save_fig_name= img_dir+patient_id+'_'+str(acquisition_date).replace(' ', '_')+'_12leads_b.png')
            
        processed += 1
        processed_files.append([xml_file, patient_id+'_'+str(acquisition_date).replace(' ', '_')+'_04leads.png'])
        
    print(f'Processed {processed} ECG files')
    ## save the processed files list
    processed_files_df = pd.DataFrame(processed_files, columns=['xml_file', 'img_file'])
    processed_files_df.to_csv(f'{img_dir}processed_files_batch{str(batch_num)}.csv', index=False)
    return processed_files


if __name__ == '__main__':
    
    masterlistcsvFile = '/workspace/ECGXML/uncertain_files_fullpath445.csv'
    IMAGE_DIR = '/workspace/ecgimgs/UnCertain/'# Directory containing images
    processed_files = fun_image_whole_folder(UncertainMasterlistcsv=masterlistcsvFile, batch=batch_num, img_dir=IMAGE_DIR, num_plots_perbatch=num_plots_perbatch, plot_12leads = False)
    print('Done generate images for batch:', batch_num)
    
    
    
    # ## create a full csv file with all files in under the Uncertain folder
    # import pandas as pd
    # import os 
    # from glob import glob
    # UncertainMasterlistcsv = '/workspace/ECGXML/uncertain_files_fullpathlrbbb.csv'
    # xml_files = [y for x in os.walk('/workspace/ECGXML/UncertainFiles/uncertains_llm_lrbbb') for y in glob(os.path.join(x[0], '*.xml'))]
    # xml_files.sort()
    # pd.DataFrame(xml_files, columns=['xml_file']).to_csv(UncertainMasterlistcsv, index=False, header=False)
    


