import numpy as np
import matplotlib.pyplot as plt


# Plot the feature importances of the forest
def plot_importance_barh(sorted_feature_name, sorted_importances, save_name, font_size):


    n_features = len(sorted_feature_name)
    #plt.barh(range(n_features), sorted(forest.feature_importances_), align='center')
    #plt.yticks(np.arange(n_features), feature_name)
    plt.barh(range(n_features), sorted_importances, align='center')
    plt.yticks(np.arange(n_features), sorted_feature_name)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.xlim(0, 0.01)
    plt.ylim(-1, n_features)
    plt.legend(loc='best').set_visible(False)
    plt.rc('font', size=font_size) 
    #plt.savefig("./rf_feature_importance.png")
    plt.savefig(save_name, bbox_inches='tight')
    plt.clf()
    #plt.show()



def plot_importance_hline(header, scores, save_name, font_size):

    # .. Code from Song MJ 2021.10

    ### Sort
    #score_float = sorted(scores.tolist(), key=float, reverse=True)
    ##print ("sort", score_float, type(scoreslist), index)
    #score_num=list()
    #for i in list(range(len(score_float))):
    # for j in list(range(len(scores))):
    #  if (score_float[i] == scores[j]):
    #    score_num.append(j)
    #  if (len(scores) == len(score_num)):
    #    break
    #new_header = [ header[i] for i in score_num ]

    new_header = header
    score_float = scores

    # Bar condition
    bar_width=0.15; opacity=0.9

    plt.figure(figsize=(10,9))
    if (len(header) == len(scores)):
     plt.hlines(y=np.nanmean(scores), xmin=0, xmax=len(header)-1, color='m', linestyles='dashed')
    else:
     plt.hlines(y=header[-1], xmin=0, xmax=len(header)-1, color='m', linestyles='dashed')
     del[header[-1]]
    plt.xticks(range(len(new_header)), new_header, rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.title('Feature_importance')
    plt.xlabel('Variables', fontsize=10, labelpad=5.0)
    plt.ylabel('Scores',fontsize=10, labelpad=6.0)
    plt.scatter(range(len(score_float)), score_float, c='r', marker='o')
    plt.bar(range(len(score_float)), score_float, bar_width, color='green', alpha=opacity, align='center')
    plt.rc('font', size=font_size) 
    plt.savefig(save_name, dpi=600)
    #plt.show()


