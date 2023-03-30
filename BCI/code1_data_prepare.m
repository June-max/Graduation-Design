% 对数据进行预处理后，根据运动想象任务提取EEG样本，构造深度学习输入样本集
clc;
clear all;
close all;
% 以下文件路径需要修改为你电脑上的文件路径
%1. 预处理好的EEG文件夹
EEGPath = 'C:\Users\Philip\Desktop\20230220BCIdatasets2a\eeg_clean_new';
INPUTPath = 'C:\Users\Philip\Desktop\20230220BCIdatasets2a\INPUT\';
Feature_all_subject = [];
Label_all_subject = [];
% 共有9笔被试的四类运动想象数据
for ID = 1 : 9
    eegname = ['A0', num2str(ID), 'T.set'];
    % 导入数据
    EEG = pop_loadset('filename', eegname, 'filepath', EEGPath);
    EEG = eeg_checkset(EEG);
    Fs = EEG.srate;% 数据采样率
    % 对数据标签进行处理，分别提取四种运动想象的标签 769 770 771 772
    mi_label = [];
    for loop = 1 : length(EEG.urevent)
        mi_label(loop, 1) = EEG.urevent(loop).type;
    end
    event = find((mi_label == 769)|(mi_label == 770)|(mi_label == 771)|(mi_label == 772));
    event_1023 = find(mi_label == 1023);
    event_temp = [];
    for cut = 1 : length(event_1023)
        event_temp = [event_temp find(event == event_1023(cut) + 1)];
    end
    mi_label(event_1023 + 1) = [];
    indexs = [];
    indexs.event1 = find(mi_label == 769);
    indexs.event2 = find(mi_label == 770);
    indexs.event3 = find(mi_label == 771);
    indexs.event4 = find(mi_label == 772);
    label_subject = zeros(max([indexs.event1;indexs.event2;indexs.event3;indexs.event4]), 1);
    label_subject(indexs.event1) = 1;
    label_subject(indexs.event2) = 2;
    label_subject(indexs.event3) = 3;
    label_subject(indexs.event4) = 4;
    label_subject(find(label_subject == 0)) = [];
    % FBCSP-空域特征提取
    % 提取不同子频段的脑电分别进行共空间模式特征提取
    Filter_band = [4, 8;8, 12;12, 16;16, 20;20, 24;24, 28;28, 32];
    Feature_subject = [];
    for loop = 1 : length(Filter_band)
        EEG = pop_loadset('filename', eegname, 'filepath', EEGPath);
        EEG = eeg_checkset(EEG);
        EEG = pop_eegfiltnew(EEG, 'locutoff', Filter_band(loop, 1), ...
            'hicutoff', Filter_band(loop, 2), 'plotfreqz', 0);
        EEG.data( : , : , event_temp) = [];
        data_subject_all = [];
        eeg_all1 = [];
        eeg_all2 = [];
        eeg_all3 = [];
        eeg_all4 = [];
        % 分别提取四种运动想象的的数据
        for trial_id = 1 : size(EEG.data, 3)
            data_med = EEG.data( : , : , trial_id)';
            % 提取每个试次任务提示出现后的4S数据作为单词运动想象EEG样本
            data_med = data_med((1.5 * Fs + 1) : 5.5 * Fs, : )';
            data_subject_all(trial_id, : , : ) = data_med;
            switch label_subject(trial_id)
                case 1
                    eeg_all1 = cat(3, eeg_all1, data_med);
                case 2
                    eeg_all2 = cat(3, eeg_all2, data_med);
                case 3
                    eeg_all3 = cat(3, eeg_all3, data_med);
                case 4
                    eeg_all4 = cat(3, eeg_all4, data_med);
            end
        end
        eeg_epoch_all = cat(3, eeg_all1, eeg_all2, eeg_all3, eeg_all4);
        EPO = [];
        cut_range = [1.5, 5.5];
        EPO{1} = reshape(eeg_all1, [22, size(eeg_all1, 2) * size(eeg_all1, 3)])';
        EPO{2} = reshape(eeg_all2, [22, size(eeg_all2, 2) * size(eeg_all2, 3)])';
        EPO{3} = reshape(eeg_all3, [22, size(eeg_all3, 2) * size(eeg_all3, 3)])';
        EPO{4} = reshape(eeg_all4, [22, size(eeg_all4, 2) * size(eeg_all4, 3)])';
        X = feature_computing(EPO, Fs);
        feature_csp = [vertcat(X{1}, X{2}, X{3}, X{4})];
        class_1_target = ones(length(X{1}), 1);
        class_2_target = 2 * ones(length(X{2}), 1);
        class_3_target = 3 * ones(length(X{3}), 1);
        class_4_target = 4 * ones(length(X{4}), 1);
        Y_train_csp = vertcat(class_1_target, class_2_target, class_3_target, class_4_target);
        Feature_subject = [Feature_subject feature_csp];
    end
    Feature_all_subject = [Feature_all_subject;Feature_subject];
    Label_all_subject = [Label_all_subject;Y_train_csp];
end
%% 保存运动想象EEG数据样本集和任务标签
Feature_all_subject_norm = [];
for i = 1 : size(Feature_all_subject, 2)
Feature_all_subject_norm( : , i) = mapminmax(Feature_all_subject( : , i)');
end
label = Label_all_subject - 1;
save(['Feature_all_subject1.mat'], 'Feature_all_subject_norm');
save Label_all_subject.txt -ascii label;