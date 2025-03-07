"""
This script is used to read ECG data from a XML file and plot the ECG data 4 leads / 12 leads.
"""

import os
import array
import base64
import xmltodict
import os
from math import ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator

# __adapted__ = "Maggie"
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
        




def add_important_points(important_points=None):
    '''
    Decorator to add important points to a 4 or 5 leads plot,
    deciding the color by the importance value.
    '''
    if important_points is None:
        important_points = []
        
    def decide_color(importance):
        # Ensure importance is between -1 and 1
        importance = max(-1, min(1, importance))
        if importance >= 0:
            # Gradient from light green (#beffbe) at 0 to dark green (#008000) at 1
            r = int(190 + (0 - 190) * importance)
            g = int(255 + (128 - 255) * importance)
            b = int(190 + (0 - 190) * importance)
        else:
            # Gradient from salmon (#fa8072) at 0 to dark red (#8b0000) at -1
            importance = abs(importance)
            r = int(250 + (139 - 250) * importance)
            g = int(128 + (0 - 128) * importance)
            b = int(114 + (0 - 114) * importance)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def decorator(plot_func):
        def wrapper(*args, **kwargs):
            fig, ax = plot_func(*args, **kwargs)
            y_min, y_max = ax.get_ylim()
            for point, importance in important_points:
                color = decide_color(importance)
                # Clip alpha between 0 and 1
                alpha = abs(importance)*0.5
                # Add a vertical span (band) at the influential point (with a small width)
                ax.axvspan(point - 1.5, point + 1.5, ymin=y_min, ymax=y_max, color=color, alpha=alpha)
            return fig, ax
        return wrapper
    return decorator




def ecg_plot_4leads(arr, 
                    title = 'ECG plot', 
                    diagnosis_note = None, 
                    whole_lead_present = ['II','V1'],
                    row_height = 6,
                    sample_rate = 250,
                    save_fig = False,
                    save_fig_name = 'output.png',
                    show_fig = True):
    
        
    # Assuming df is your nd array with ECG data
    # Columns: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # for sample rate = 250: 250 small boxes, 50 large boxes, one large box = 50
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    lead_to_idx = {name: i for i, name in enumerate(lead_names)}
    
    color_major = (1, 0, 0)
    color_minor = (1, 0.7, 0.7)

    leadpoints = arr.shape[1]
    if leadpoints > sample_rate * 10:
        factor = leadpoints // (sample_rate * 10)
        arr = arr[:, ::factor]
        leadpoints = arr.shape[1]
        
    major_grid = 50
    x_max, x_min = sample_rate * 10, -50
    secs = 10
    rows = 3 + len(whole_lead_present)
    
    # Define the groups for each subplot
    groups = [
        ['I', 'aVR', 'V1', 'V4'],
        ['II', 'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6'],
        *whole_lead_present
    ]
    # Scale signal amplitude.
    arr = arr / 2.0
    
    # Calculate extra_y_offset using the whole_lead_present group signal
    # Get the minimum value among the signals of the last group (if exists)
    if whole_lead_present:
        last_idx = lead_to_idx[whole_lead_present[-1]]
        min_val = arr[last_idx].min()
    else:
        min_val = 0
    extra_y_offset = int(-100 - (min_val // major_grid) * major_grid) if (min_val // major_grid) * major_grid < -100 else 0
           
    y_max = 200
    y_min = 200 - rows * major_grid * row_height - extra_y_offset

    fig, ax = plt.subplots(figsize=(secs*5, row_height*rows + extra_y_offset/major_grid), subplot_kw={'xticks': [], 'yticks': []}, dpi=150)

    for i, group in enumerate(groups):
        y_offset = 200 - i * major_grid * row_height
        if i < 3:
            # For the first three groups, concatenate segmented parts of each lead signal.
            concatenated_signal = np.array([])
            current_position = 0
            for j, lead in enumerate(group):
                lead_signal = arr[lead_to_idx[lead], :]
                quarter_length = lead_signal.shape[0] // 4
                start = j * quarter_length
                end = (j + 1) * quarter_length
                seg = lead_signal[start:end]
                if concatenated_signal.size == 0:
                    concatenated_signal = seg
                else:
                    concatenated_signal = np.concatenate((concatenated_signal, seg))
                ax.text(current_position + 5, y_offset - 250, lead, 
                        verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=8)
                current_position += seg.size
                ax.plot([current_position, current_position], [y_offset - 230, y_offset - 170], 
                        color='black', linewidth=0.8)
            x_vals = np.arange(concatenated_signal.size)
            ax.plot(x_vals, concatenated_signal + y_offset - 200, linewidth=1, color='black')
        else:
            # For whole lead present groups (single lead plotting)
            lead = group  # group is a string here
            lead_signal = arr[lead_to_idx[lead], :]
            x_vals = np.arange(lead_signal.shape[0])
            ax.plot(x_vals, lead_signal + y_offset - 200, linewidth=1, color='black')
            ax.text(5, y_offset - 250, lead, verticalalignment='bottom', horizontalalignment='left', 
                    color='black', fontsize=8)
        
        # Add horizontal line at y=100 from x=-50 to x=0 for each subplot.
        ax.plot([-50, 0], [y_offset - 100, y_offset - 100], color='black', linewidth=1)
        # Add vertical line at x=0 from y=0 to y=100.
        ax.plot([0, 0], [y_offset - 200, y_offset - 100], color='black', linewidth=1)
        # Add vertical line at x=-50 from y=0 to y=100.
        ax.plot([-50, -50], [y_offset - 200, y_offset - 100], color='black', linewidth=1)

        
    ax.set_xticks(np.arange(x_min,x_max,50))    
    ax.set_yticks(np.arange(y_min,y_max,50))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which='major', linestyle='-', linewidth=0.5, color=color_major, alpha=0.5)
    ax.grid(which='minor', linestyle='-', linewidth=0.5 , color=color_minor, alpha=0.5)
    
    # Remove numbers along x and y axis
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
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
    elif show_fig:
        plt.show()
    return fig, ax
    
    
    

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
    ecg        : ECG data in numpy array format (nleads x leadpoints).
    sample_rate: Sample rate of the signal.
    title      : Title which will be shown on top of chart.
    columns    : Number of columns for display.
    row_height : Height of each row/grid.
    show_lead_name : Whether to show the lead name.
    show_grid      : Whether to show grid lines.
    show_separate_line  : Whether to show separation lines between columns.
    """
    assert isinstance(ecg, np.ndarray) and ecg.shape[0] == 6, "ecg must be a numpy array and shape (6, n)"
    # If the ECG is too long, downsample along the time axis
    if ecg.shape[1] > sample_rate * 10:
        downsampling = ecg.shape[1] // (sample_rate * 10)
        ecg = ecg[:, ::downsampling]
        
    nleads = ecg.shape[0]
    # lead_index = [f"Lead {i+1}" for i in range(nleads)]
    lead_index = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    # Normalize the ECG data for plotting
    ecg_to_plot = ecg / 200 + 1

    lead_order = list(range(nleads))
    secs  = 10
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    
    line_width = 1
    fig, ax = plt.subplots(figsize=(secs * columns , rows * row_height / 5 ), subplot_kw={'xticks': [], 'yticks': [], 'frame_on': False}, dpi=150)
    fig.suptitle(title)

    x_min = -0.2
    x_max = columns * secs
    y_min = row_height/2 - rows*(row_height/2)
    y_max = row_height/2

    if style == 'bw':
        color_major = (0.4, 0.4, 0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0, 0, 0)
    else:
        color_major = (1, 0, 0)
        color_minor = (1, 0.7, 0.7)
        color_line  = (0, 0, 0.7)

    if show_grid:
        ax.set_xticks(np.arange(x_min, x_max, 0.2))    
        ax.set_yticks(np.arange(y_min, y_max, 0.5))
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which='major', linestyle='-', linewidth=0.5, color=color_major, alpha=0.5)
        ax.grid(which='minor', linestyle='-', linewidth=0.5, color=color_minor, alpha=0.5)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    # Plot each lead in its respective subplot position
    for c in range(columns):
        for i in range(rows):
            idx = c * rows + i
            if idx < leads:
                # y_offset determines vertical placement for this lead
                y_offset = -(row_height/2) * i
                x_offset = secs * c if c > 0 else 0
                if c > 0 and show_separate_line:
                    ax.plot([x_offset, x_offset], [y_offset - 0.3, y_offset + 0.3], linewidth=line_width, color=color_line)
                t_lead = lead_order[idx]
        
                step = 1.0 / sample_rate
                if show_lead_name:
                    ax.text(x_offset + 0.07, y_offset + 0.1, lead_index[t_lead], fontsize=10)
                x_vals = np.arange(ecg_to_plot[t_lead].shape[0]) * step + x_offset
                ax.plot(x_vals, ecg_to_plot[t_lead] + y_offset, linewidth=line_width, color=color_line)
                # Add horizontal line at y=y_offset+2 from x_offset-0.2 to x_offset
                ax.plot([x_offset - 0.2, x_offset], [y_offset + 2, y_offset + 2], color=color_line, linewidth=line_width )
                # Add vertical lines at x=x_offset and x=x_offset-0.2 from y_offset+1 to y_offset+2
                ax.plot([x_offset, x_offset], [y_offset + 1, y_offset + 2], color=color_line, linewidth=line_width)
                ax.plot([x_offset - 0.2, x_offset - 0.2], [y_offset + 1, y_offset + 2], color=color_line, linewidth=line_width)
        
    fig.set_size_inches(16, 9)
    fig.subplots_adjust(
        hspace=0, 
        wspace=0,
        left=0.109375, 
        right=1 - 0.109375,  
        bottom=0,
        top=1
    )
    
    if save_fig:
        plt.savefig(save_fig_name, dpi=150)
        plt.close()
    else:
        plt.show()
    return fig, ax




if __name__ == '__main__':
    
    # ## loop through all the xml files in the dataxml folder
    # for xml_file in glob.glob('ECGXML/ECGXML/*.xml'):
    #     ecg = ECGXMLReader(xml_file)
    
    cur_file = '/workspace/ECGXML/CertainFiles2502/MUSE_20240704_112553_40000.xml'
    img_dir = '/workspace/ecgimgs'
    ecg = ECGXMLReader(cur_file, augmentLeads=True)
    patient_id = ecg.patientId
    diagnosis = ecg.diagnosis
    
    specified_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    all_voltages = ecg.getAllVoltages()
    ordered_voltages = {lead: all_voltages[lead] for lead in specified_order if lead in all_voltages}
    ecgarr = np.array(list(ordered_voltages.values())).astype('float32')
    

    # df = pd.DataFrame(ecg.getAllVoltages(),columns=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
    fig, ax = ecg_plot_4leads(ecgarr, title = 'previous ECG 4 - Patient ID: ' + patient_id, diagnosis_note=diagnosis, save_fig=False, save_fig_name=img_dir+'/'+patient_id+'_04leads.png')
    
    
    ecg_plot12(ecgarr[0:6, :], sample_rate = 250, title = 'ECG 12a - ID: ' + patient_id, save_fig=False, save_fig_name= img_dir+'/'+patient_id+'_12leads_a.png')

    ecg_plot12(ecgarr[6:12, :], sample_rate = 250, title = 'ECG 12b - ID: ' + patient_id,save_fig=False, save_fig_name=img_dir+'/'+patient_id+'_12leads_b.png')
    
    
    

