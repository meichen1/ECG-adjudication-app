"""
This script is used to read ECG data from a XML file and plot the ECG data 4 leads / 12 leads.
"""

import os
import array
import base64
import xmltodict
import os
from math import ceil, floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# __author__ = "Will Hewitt"
# __credits__ = ["Will Hewitt"]
# __version__ = "1.0.0"
# __maintainer__ = "Will Hewitt"
# __email__ = "me@hewittwill.com"
# __status__ = "Development"

class ECGXMLReader:
    
    """ Extract voltage data from a ECG XML file """
    
    def __init__(self, path, augmentLeads=True, verbose=False):
        
        if verbose:
            print('Current xml:', path)
        try:
            with open(path, 'rb') as xml:
                self.ECG = xmltodict.parse(xml.read().decode('utf8'))
            
            self.filename               = os.path.basename(path)
            self.augmentLeads           = augmentLeads
            self.path                   = path
            self.patientId              = self.ECG['RestingECG']['PatientDemographics']['PatientID']

            self.PatientDemographics    = self.ECG['RestingECG']['PatientDemographics']
            self.ECGtime                = self.ECG['RestingECG']['TestDemographics']['AcquisitionDate']+ ' ' + self.ECG['RestingECG']['TestDemographics']['AcquisitionTime']
            self.TestDemographics       = self.ECG['RestingECG']['TestDemographics']
            self.RestingECGMeasurements = self.ECG['RestingECG']['RestingECGMeasurements']
            self.Waveforms              = self.ECG['RestingECG']['Waveform'][1]
            self.ECGSampleBase          = int(self.Waveforms['SampleBase'])
            #rhythm
            self.LeadVoltages           = None
            
            
            self.OrderingMDName         = self.ECG['RestingECG']['TestDemographics']['OrderingMDFirstName'] if 'OrderingMDFirstName' in self.ECG['RestingECG']['TestDemographics'] else ''
            self.OrderingMDName     = self.OrderingMDName + (self.ECG['RestingECG']['TestDemographics']['OrderingMDLastName'] if 'OrderingMDLastName' in self.ECG['RestingECG']['TestDemographics'] else '')
            
            self.OverReaderMDName       = self.ECG['RestingECG']['TestDemographics']['OverReaderFirstName']+' '+self.ECG['RestingECG']['TestDemographics']['OverreaderLastName'] if 'OverReaderFirstName' in self.ECG['RestingECG']['TestDemographics'] else ''
            
            self.EditorName             = self.ECG['RestingECG']['TestDemographics']['EditorFirstName']+' '+self.ECG['RestingECG']['TestDemographics']['EditorLastName'] if 'EditorFirstName' in self.ECG['RestingECG']['TestDemographics'] else ''
            
            if 'Order' in self.ECG['RestingECG']:
                self.AttendingMDName         = self.ECG['RestingECG']['Order']['AttendingMDFirstName']+' '+ self.ECG['RestingECG']['Order']['AttendingMDLastName'] if 'AttendingMDFirstName' in self.ECG['RestingECG']['Order'] else ''
                
                self.AdmittingMDName         = self.ECG['RestingECG']['Order']['AdmittingMDFirstName']+' '+ self.ECG['RestingECG']['Order']['AdmittingMDLastName'] if 'AdmittingMDFirstName' in self.ECG['RestingECG']['Order'] else ''
            else:
                self.AttendingMDName         = ''
                self.AdmittingMDName         = ''
                
                
            self.diagnosis = ''    
            DiagStatement = self.ECG['RestingECG']['Diagnosis']['DiagnosisStatement']
            
            
            if type(DiagStatement) == list:
                self.diagnosis = '\n'.join(filter(None, [textDic['StmtText'] for textDic in DiagStatement]))
            else:
                self.diagnosis = DiagStatement['StmtText'] if DiagStatement is not None else ''
            
        except Exception as e:
            print(str(e))
    
        
    def makeLeadVoltages(self):
        
        ## lead names: ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        leads = {}
        num_leads = 0

        for lead in self.Waveforms['LeadData']:
            num_leads += 1
            
            lead_data = lead['WaveFormData']
            lead_b64  = base64.b64decode(lead_data)
            lead_vals = np.array(array.array('h', lead_b64))
            leads[ lead['LeadID'] ] = lead_vals
            
        if num_leads > 8 and not self.augmentLeads:
            leadsID8 = ('I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
            # Iterate over a copy of the keys so we can delete from the original dictionary
            for lead in list(leads.keys()):
                if lead not in leadsID8:
                    del leads[lead]
        
        if num_leads == 8 and self.augmentLeads:

            leads['III'] = np.subtract(leads['II'], leads['I'])
            leads['aVR'] = np.add(leads['I'], leads['II'])*(-0.5)
            leads['aVL'] = np.subtract(leads['I'], 0.5*leads['II'])
            leads['aVF'] = np.subtract(leads['II'], 0.5*leads['I'])
            
        elif num_leads == 11 and self.augmentLeads:
            leadsID8 = ('I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
            # Iterate over a copy of the keys so we can delete from the original dictionary
            for lead in list(leads.keys()):
                if lead not in leadsID8:
                    del leads[lead]
            
            leads['III'] = np.subtract(leads['II'], leads['I'])
            leads['aVR'] = np.add(leads['I'], leads['II'])*(-0.5)
            leads['aVL'] = np.subtract(leads['I'], 0.5*leads['II'])
            leads['aVF'] = np.subtract(leads['II'], 0.5*leads['I'])
            
        return leads
    

    def getLeadVoltages(self, LeadID):
        self.LeadVoltages = self.makeLeadVoltages()
        return self.LeadVoltages[LeadID]
    
    def getAllVoltages(self):
        self.LeadVoltages = self.makeLeadVoltages()
        return self.LeadVoltages
    
    def visualize_ecg(self):
        
        self.makeLeadVoltages()
        num_leads = len(self.LeadVoltages)
        fig, axes = plt.subplots(num_leads, 1, figsize=(10, 18))
        fig.suptitle(f"ECG Signals for Patient '{self.patientId}'")
        
        ## add textbox  of the plot and auto word wrap for diagnosis and avoid overlap with the plot below
        
        fig.text(0.91, 0.91, f"Diagnosis: {self.diagnosis}", ha='center', va='center',wrap=True)
        
        for i, (lead_name, signal) in enumerate(self.LeadVoltages.items()):
            axes[i].plot(signal)
            axes[i].set_title(f'{lead_name}')  
            ## set the axis box to be alpha 0.5
            for spine in axes[i].spines.values():
                spine.set_alpha(0.3)
        
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.96])
        plt.show()



# Assuming df is your DataFrame with ECG data
# Columns: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# 250 small boxes, 50 large boxes, one large box = 50

def ecg_plot_4leads(df, 
                    title = 'ECG 4', 
                    diagnosis_note = None, 
                    save_fig = False,
                    save_fig_name = 'output.png',
                    whole_lead_present = ['II','V1'],
                    row_height = 6,
                    sample_rate = 250,
                    show_frame = False
                    ):
    
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
    
    if df.shape[0] > 2500:
        sampling_rate = df.shape[0]//2500
        df = df.iloc[::sampling_rate]
    
    # Define the groups for each subplot
    groups = [
        ['I', 'aVR', 'V1', 'V4'],
        ['II', 'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6'],
        *whole_lead_present  # For the last subplot, only 'V1' for all points
    ]
    df = df / 2
    
    major_grid = 50
    # df_max = ceil(df.max().max()/50)*50
    # df_min = floor(df.min().min()/50)*50
    # y_max = df_max + 50
    # y_min = df_min - 50
    
    x_max = df.shape[0]
    x_min = -50
    secs = x_max / sample_rate
    rows = 3 + len(whole_lead_present)
    
    extra_y_offset = int(-100 - df[groups[-1]].min() // major_grid * major_grid) if df[groups[-1]].min() // major_grid * major_grid < -100 else 0
           
    y_max = 200
    y_min = 200 - rows * major_grid * row_height - extra_y_offset
    
    ## scale y-axis from 0 to major_grid * row_height (300); six major grid lines
    # df = (df - df_min)/(df_max - df_min) * major_grid * row_height
    
    # Create subplots
    fig, ax = plt.subplots(figsize=(secs*5, row_height*rows+ extra_y_offset//major_grid ), subplot_kw={'xticks': [], 'yticks': [], 'frame_on': show_frame}, dpi=150)

    for i, group in enumerate(groups):
        
        y_offset = 200 - i * major_grid * row_height
        if i < 3:  # For the first three subplots
            concatenated_series = pd.Series(dtype='float32')
            current_position = 0
            for j, lead in enumerate(group):
                quarter_length = len(df[lead]) // 4
                start = j * quarter_length
                end = (j + 1) * quarter_length
                concatenated_series = pd.concat([concatenated_series, df[lead].iloc[start:end]])
                # Add lead name text annotation
                ax.text(current_position + 5, y_offset-250, lead, verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=8)
                
                current_position += quarter_length  # Update current position
                ax.plot([current_position,current_position], [y_offset-200-30, y_offset-200+30] ,color='black', linewidth=0.8)
                
            ax.plot(concatenated_series.reset_index(drop=True) + y_offset-200, linewidth=1, color='black')
            
        elif i >= 3:
            
            ax.plot(df[groups[i]]+ y_offset -200, linewidth=1, color='black')
            # Add lead name text annotation for whole_lead_present
            ax.text(5, y_offset-250, groups[i], verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=8)
        
        # Add horizontal line at y=100 from x=-50 to x=0
        ax.plot([-50, 0], [y_offset-100, y_offset-100], color='black', linewidth=1)
        
        # Add vertical line at x=0 from y=0 to y=100
        ax.plot([0, 0], [y_offset-200, y_offset-100], color='black', linewidth=1)
        
        # Add vertical line at x=-50 from y=0 to y=100
        ax.plot([-50, -50], [y_offset-200, y_offset-100], color='black', linewidth=1)
        
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
        
    # Add custom grids
    for x in range(x_min, x_max+10, 10):  # Every 10 points
        alpha, linewidth = (1, 0.2) if x % 50 == 0 else (0.6, 0.1)
        ax.axvline(x=x, color='red', alpha=alpha, linewidth=linewidth)
    for y in range(y_min, y_max+10, 10):  # Every 50 points
        alpha, linewidth = (1, 0.2) if y % 50 == 0 else (0.6, 0.1)
        ax.axhline(y=y, color='red', alpha=alpha, linewidth=linewidth)
    
    # Force the grid to be square
    ax.set_aspect('equal', adjustable='box')

    # Remove newline characters from diagnosis_note
   
    if diagnosis_note:
        diagnosis_note_cleaned = diagnosis_note.replace('\n', '; ')
        ax.text(0.5, 0.2, diagnosis_note_cleaned, transform=ax.transAxes, fontsize=8, color='black', alpha=0.5, ha='center', va='center')
    
    if title:
        fig.suptitle(title)
        
    ## resize this plot to have 16:9 aspect ratio by adding white space to the left and right
    fig.set_size_inches(16, 9)    
    # Adjust the spacing of the subplots
    plt.subplots_adjust(left=1/32, right=1-1/32, bottom=0,  top=1, wspace=0, hspace=0)
  
    if save_fig:
        plt.savefig(save_fig_name, dpi=150)
        plt.close()
    else:
        plt.show()
    
    


    
def ecg_plot12(
        ecg, 
        sample_rate    = 250, 
        title          = 'ECG 12', 
        save_fig       = False,
        save_fig_name  = 'output12.png',
        columns        = 1,
        row_height     = 6,
        style          = None,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line  = True,
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : ECG data in pd.DataFrame format, each column is a lead signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        columns    : display columns, defaults to 2
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """
    assert isinstance(ecg, pd.DataFrame), "df must be a pandas DataFrame"
    
    if ecg.shape[0] > 2500:
        sampling_rate = ecg.shape[0]//2500
        ecg = ecg.iloc[::sampling_rate]
    
    lead_index = list(ecg.columns)
    # df_max = ceil(ecg.max().max()/50)*50
    # df_min = floor(ecg.min().min()/50)*50
    
    # ecg = (ecg - df_min)/(df_max - df_min + 100)*3
    ecg = ecg / 200 + 1
    ecg = ecg.values.T
    
    lead_order = list(range(0,len(lead_index)))
    secs  = len(ecg[0])/sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    
    display_factor = 1  # 2.5
    line_width = 1
    fig, ax = plt.subplots(figsize=(secs * columns * display_factor, rows * row_height / 5 * display_factor), subplot_kw={'xticks': [], 'yticks': [], 'frame_on': False}, dpi=150)
    fig.suptitle(title)

    x_min = -0.2
    x_max = columns* secs
    y_min = row_height/2 - rows*(row_height/2)
    y_max = row_height/2

    if (style == 'bw'):
        color_major = (0.4, 0.4, 0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    else:
        color_major = (1, 0, 0)
        color_minor = (1, 0.7, 0.7)
        color_line  = (0, 0, 0.7)

    if (show_grid):
        ax.set_xticks(np.arange(x_min,x_max,0.2))    
        ax.set_yticks(np.arange(y_min,y_max,0.5))

        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major,alpha=0.5)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor,alpha=0.5)
        
    # Remove numbers along x and y axis
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)


    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height/2) * ceil(i%rows)
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25
                x_offset = 0
                if(c > 0):
                    x_offset = secs * c
                    if(show_separate_line):
                        ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line)

                t_lead = lead_order[c * rows + i]
         
                step = 1.0/sample_rate
                if(show_lead_name):
                    ax.text(x_offset + 0.07, y_offset + 0.1, lead_index[t_lead], fontsize= 10 * display_factor)
                ax.plot(
                    np.arange(0, len(ecg[t_lead])*step, step) + x_offset, 
                    ecg[t_lead] + y_offset,
                    linewidth=line_width * display_factor, 
                    color=color_line
                    )
                # Add horizontal line at y=1 from x=-0.2 to x=0
                ax.plot([x_offset - 0.2, x_offset], [y_offset + 2, y_offset + 2], color=color_line, linewidth=line_width * display_factor)
                
                # Add vertical line at x=0 from y=0 to y=1
                ax.plot([x_offset, x_offset], [y_offset+1, y_offset + 2], color=color_line, linewidth=line_width * display_factor)
                
                # Add vertical line at x=-0.2 from y=0 to y=1
                ax.plot([x_offset - 0.2, x_offset - 0.2], [y_offset+1, y_offset + 2], color=color_line, linewidth=line_width * display_factor)
                
        ## resize this plot to have 16:9 aspect ratio by adding white space to the left and right
        fig.set_size_inches(16, 9)
         # display_factor = display_factor ** 0.5
        fig.subplots_adjust(
            hspace = 0, 
            wspace = 0,
            left   = 0.109375, 
            right  = 1-0.109375,  
            bottom = 0,  # the bottom of the subplots of the figure
            top    = 1
        )
        # fig.tight_layout()
        
        if save_fig:
            plt.savefig(save_fig_name, dpi=150)
            plt.close()
        else:
            plt.show()



if __name__ == '__main__':
    
    # ## loop through all the xml files in the dataxml folder
    # for xml_file in glob.glob('ECGXML/ECGXML/*.xml'):
    #     ecg = ECGXMLReader(xml_file)
        
    #     print(ecg.patientId)
    #     print(ecg.diagnosis)
    #     print(ecg.diagTwoClass)
    #     print('----------------------------------------')
          
    cur_file = '/workspace/ECGXML/WCT-Exports-Full-2407/20240703-Antiperovich/nonVT/MUSE_20240703_082302_39000.xml'
    img_dir = '/workspace/ecgimgs'
    ecg = ECGXMLReader(cur_file, augmentLeads=True)
    patient_id = ecg.patientId
    diagnosis = ecg.diagnosis

    df = pd.DataFrame(ecg.getAllVoltages(),columns=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

    ecg_plot_4leads(df, title = 'previous ECG 4 - Patient ID: ' + patient_id, diagnosis_note=diagnosis, save_fig=False, save_fig_name=img_dir+'/'+patient_id+'_04leads.png')

    ecg_plot12(df.iloc[:,0:6], sample_rate = 250, title = 'ECG 12a - ID: ' + patient_id, save_fig=False, save_fig_name= img_dir+'/'+patient_id+'_12leads_a.png')

    ecg_plot12(df.iloc[:,6:12], sample_rate = 250, title = 'ECG 12b - ID: ' + patient_id,save_fig=False, save_fig_name=img_dir+'/'+patient_id+'_12leads_b.png')
    


