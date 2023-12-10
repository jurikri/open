#%% frontiers - Fig2 - 
# Re-importing necessary libraries since the execution state was reset
import numpy as np
import matplotlib.pyplot as plt

# Load the data again
# X = np.load('/mnt/data/X.npy')
# Y = np.load('/mnt/data/Y.npy')

# Define the feature data and class labels
fn = 12
for fn in [0, 1, 2, 3, 12]:
    #%%
    feature_data = X[:, fn]
    labels = ['Nonpain', 'Pain', 'Morphine', 'Ketoprofen']
    custom_colors = ['gray',  'red', (0.461, 0.566, 0.897), (0.484, 0.737, 0.897)]
    
    # Create the plot with transparent background
    fig, ax = plt.subplots(figsize=(2.5/2, 2/2), dpi=300)    
    msr = np.mean([2.5/2/8, 2/2/6])
    
    if fn == 12:
        fig, ax = plt.subplots(figsize=(2.2, 1.98), dpi=300)    
        msr = np.mean([2.5/8, 2/6])
    
    # Data for the violin plot
    data_for_box = [feature_data[Y[:, j] == 1] for j in range(Y.shape[1])]
    
    # Create violin plot
    positions = np.arange(1, len(data_for_box) + 1) * 0.75
    violin = ax.violinplot(data_for_box, showmeans=False, \
                           showmedians=False, showextrema=False, widths=0.65, positions=positions)
    
    # 바이올린 플롯의 테두리 제거
    for pc in violin['bodies']:
        pc.set_linewidth(0)  # 테두리 두께를 0으로 설정하여 테두리 제거

    # Customizing the violin plot colors
    for i, pc in enumerate(violin['bodies']):
        pc.set_facecolor(custom_colors[i])
        pc.set_edgecolor(custom_colors[i])
        pc.set_alpha(0.7)
    
    # mean, error bar plot
    # Calculate means and SEMs for each class
    means = [np.mean(data) for data in data_for_box]
    sems = [np.std(data) / np.sqrt(len(data)) for data in data_for_box]
    
    for i in range(len(data_for_box)):
        mean_val = means[i]
        sem_val = sems[i]
        plt.errorbar(positions[i], mean_val, yerr=sem_val, fmt='o', color='white', \
                     ecolor='black', capsize=20*msr, capthick=2*msr, elinewidth=2*msr, \
                     markeredgecolor='black', markersize=5*msr, markeredgewidth=2*msr)
        
    line_width = 0.2
    for spine in ax.spines.values(): # 축 두깨 조절
        spine.set_linewidth(line_width) 
        
    # X축과 Y축 눈금의 두께를 설정
    ax.xaxis.set_tick_params(width=line_width)  # X축 눈금의 두께 설정
    ax.yaxis.set_tick_params(width=line_width)  # Y축 눈금의 두께 설정
    ax.yaxis.set_tick_params(pad=1)  # Y축 눈금 레이블과 눈금 사이의 간격을 5포인트로 설정

    # Set y-ticks to [1.0, 0.5, 0.0]

    # X축과 Y축 눈금 길이 설정
    ax.xaxis.set_tick_params(length=2)  # X축 눈금의 길이를 2로 설정
    ax.yaxis.set_tick_params(length=2)  # Y축 눈금의 길이를 2로 설정
    
    # 폰트 스타일 설정
    from matplotlib.font_manager import FontProperties
    font_prop = FontProperties(family='Calibri', style='normal', weight='normal', size=8)
    
    # Y축 눈금 레이블에 폰트 스타일 적용
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        
    ax.set_ylim(-0.1, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0], fontsize=8)
    ax.set_yticklabels(['0.0', '0.5', '1.0'])
    
    if fn == 12:
        ax.set_ylim(0, 1)
        ax.set_yticks(list(np.round(np.arange(0,1.01,0.2), 2)), fontsize=8)
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        
     # Remove x-tick labels and title
    ax.set_xticks(positions, ['']*len(labels))
    if fn in [0,2,1,3]: 
        ax.xaxis.set_tick_params(length=0)  # X축 눈금 길이를 0으로 설정
        for label in ax.get_xticklabels():
            label.set_color('none')  # 'none'은 투명한 색상을 의미
            
    if fn == 12:
        ax.set_xticklabels(['Nonpain', 'Pain', 'Mor', 'Keto'], fontsize=7)
        
    if fn in [2,3]:
        # Y축 눈금을 제거하지만 레이블은 유지
        ax.yaxis.set_tick_params(length=0)  # Y축 눈금 길이를 0으로 설정
        for label in ax.get_yticklabels():
            label.set_color('none')  # 'none'은 투명한 색상을 의미

    plt.tight_layout()
    save_path = r'C:\SynologyDrive\worik in progress\thesis 논문화\figure\fig_fn' + str(fn) + '.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
