clear all
close all
clc

% Loading data:

MNI_T1w = load_untouch_nii('MNI_T1ww.nii');
MNI_T1w_image = double(MNI_T1w.img);
figure, imagesc(squeeze(rot90(MNI_T1w_image(:,:,50)))), title('MRI T1w in the MNI Schaefer Atlas space')

fMRI = load_untouch_nii('patient_preproc_rs_fMRI.nii');
fMRI_image = double(fMRI.img);
figure, imagesc(squeeze(rot90(fMRI_image(:,:,50,150)))), title('fMRI in the MNI Schaefer Atlas space')

summedfMRI = load_untouch_nii('patient_brain_mask.nii');
summedfMRI_image = double(summedfMRI.img);
figure, imagesc(squeeze(rot90(summedfMRI_image(:,:,50)))), title('fMRI summed over time the MNI Schaefer Atlas space'), colorbar

Schaefer_Atlas = load_untouch_nii('MNI_Schaefer_segmentation.nii');
Schaefer_Atlas_image = double(Schaefer_Atlas.img);
figure, imagesc(squeeze(rot90(Schaefer_Atlas_image(:,:,50)))), title('Schaefer Atlas'), colorbar

[nR, nC, nS, nVol] = size(fMRI_image);

figure,

for i = 1:size(MNI_T1w_image,3)
    imagesc(rot90(squeeze(MNI_T1w_image(:,:,i)))), colorbar
    pause
end

figure, 

for i = 1:size(Schaefer_Atlas_image,3)
    imagesc(rot90(squeeze(Schaefer_Atlas_image(:,:,i)))), colorbar
    pause(0.5)
end


%% Segment MNI_T1w into GM, WM, CSF using the SPM12. 

GM_data = load_untouch_nii('c1MNI_T1ww.nii');
WM_data = load_untouch_nii('c2MNI_T1ww.nii');
CSF_data = load_untouch_nii('c3MNI_T1ww.nii');

GM = (double(GM_data.img));
WM = (double(WM_data.img));
CSF = (double(CSF_data.img));

figure,
for i = 1:size(GM,3)
    imagesc(rot90(squeeze(GM(:,:,i)))), colorbar
    pause(0.5)
end

figure,
for i = 1:size(WM,3)
    imagesc(rot90(squeeze(WM(:,:,i)))), colorbar
    pause(0.5)
end

figure,
for i = 1:size(CSF,3)
    imagesc(rot90(squeeze(CSF(:,:,i)))), colorbar
    pause(0.5)
end

%% Binarize the WM and CSF maps by using a threshold of 0.95 for WM and 0.8 for CSF. 

WM_threshold = 0.95;
CSF_threshold = 0.80;
graylevel = 255;

Binarized_WM = (WM > WM_threshold*graylevel);
Binarized_CSF = (CSF > CSF_threshold*graylevel);

maskWM = double(Binarized_WM);
maskCSF = double(Binarized_CSF);

figure,
for i = 1:size(Binarized_CSF,3)
    imagesc(rot90(squeeze(Binarized_CSF(:,:,i)))), colorbar
    pause(0.5)
end

figure,
for i = 1:size(Binarized_WM,3)
    imagesc(rot90(squeeze(Binarized_WM(:,:,i)))), colorbar
    pause(0.5)
end

%% Generate the variable WM_matrix containing the signals of all the voxelx included in the WM binarized map. 
% Perform a PCA on WM_matrix and save in the matrix WM_regressos the first 5 eigenvectors 
% (expected matrix size: n of slices x 5).

% Masking Binarized_WM with summedfMRI_image:

WM_masked = maskWM.*summedfMRI_image;
WM_index = find(WM_masked>0);

figure, imagesc(squeeze(WM_masked(:,:,45))), title('WM_masked with summedfMRI_image')

% Voxels dynamics extraction:

disp('Creating data2D matrix: WM voxels dynamics extraction')
WM_matrix = zeros([nVol, size(WM_index, 1)]);

for t = 1:nVol
    tmp = squeeze(fMRI_image(:,:,:,t));       % Volume selection
    WM_matrix(t,:)  = tmp(WM_index);          % WM Time Activity Curve extraction
end

% Performing PCA:

WM_matrix_z = zscore(WM_matrix);
[coeffWM, scoresWM] = pca(WM_matrix_z);
WM_regressors = scoresWM(:,1:5);

%% Generate the variable CSF_matrix containing the signals of all the voxels included in the 
% CSF binarized map. Perform a PCA on CSF_matrix and save in the matrix CSF_regressos 
% the first 5 eigenvectors (expected matrix size: n of volumes x 5). Calculate the first derivative 
% of each eigenvector and add this additional signals to the CSF_regressos matrix (now the 
% expected size is n of slices x 10)

% Masking Binarized_CSF with summedfMRI_image:

CSF_masked = maskCSF.*summedfMRI_image;
CSF_index = find(CSF_masked>0);

figure, imagesc(squeeze(CSF_masked(:,:,45))), title('CSF_masked with summedfMRI_image')

% Voxels dynamics extraction:

disp('Creating data2D matrix: CSF voxels dynamics extraction')
CSF_matrix = zeros([nVol, size(CSF_index, 1)]);

for t = 1:nVol
    tmp = squeeze(fMRI_image(:,:,:,t));       % Volume selection
    CSF_matrix(t,:) = tmp(CSF_index);         % CSF Time Activity Curve extraction
end

% Performing PCA:

CSF_matrix_z = zscore(CSF_matrix);
[coeffCSF, scoresCSF] = pca(CSF_matrix_z);

CSF_regressors = scoresCSF(:,1:5);
CSF_regressors_0 = [zeros(1,5); CSF_regressors];
CSF_reg_diff = diff(CSF_regressors_0);
CSF_regressors = [CSF_regressors CSF_reg_diff];

%% Generate the variable movement_matrix containing the 6 movement regressors traces and 
% their first derivatives (expected matrix size: n of volumes x 12).

movement_matrix = importdata('patient_motion_correction_param.txt');
movement_matrix_0 = [zeros(1,size(movement_matrix,2)); movement_matrix];
diff_movement = diff(movement_matrix_0);
movement_matrix = [movement_matrix diff_movement];


%% Extract for each ROI of the Schaefer atlas the average signal (excluding voxels that feature 
% a probability of belonging to WM greater than 95%).

signal = WM < WM_threshold*graylevel;
mask = Schaefer_Atlas_image.*signal;
figure, imagesc(squeeze(mask(:,:,55))), title('Masking Hammers Atlas')

disp('Masking Hammers Atlas with EPI mask')
mask = mask .* summedfMRI_image;
figure, imagesc(squeeze(mask(:,:,1))), title('Masking Hammers Atlas with summedfMRI'),colorbar

number_ROIs = max(mask(:));
ROI_TAC = zeros(nVol,number_ROIs);

for i = 1 : number_ROIs
    
    ROImask = mask==i;
    data2D  = zeros(nVol, sum(ROImask(:)));
    for t = 1: nVol
        tmp = squeeze(fMRI_image(:,:,:,t));
        data2D(t,:) = tmp(ROImask);
        clear tmp
    end
    
    ROI_TAC(:,i) = nanmean(data2D,2);

end
   

figure, plot(1:nVol,ROI_TAC(:,1:100)), title('Noised ROIs')


%% Data Denoising
% Noise Regression

disp('Regress out motion and WM and CSF')

X = [movement_matrix WM_regressors CSF_regressors];
X = zscore(X);

beta = inv(X'*X)*X'*ROI_TAC; 
noise = X*beta;
data  = ROI_TAC-X*beta;

figure, plot(1:nVol,ROI_TAC(:,1:100)), title('Noised ROIs')
figure, plot(1:nVol,noise(:,1:100)), title('Phisiological noise and motion artifacts contributions')
figure, plot(1:nVol,data(:,1:100)), title('Denoised ROIs')

%% Temporal filtering: filter the denoised ROI signals with a high pass filter with a cut-off 
% frequency of 1/128 Hz.

TR = 1.4;
frequency = 1/128; 

disp('High pass temporal filtering')
data_filtered = hp_filter(data,nVol,TR,frequency);

figure, plot(1:nVol,data(:,1:10)), title('Denoised ROIs before Temporal Filtering')
figure, plot(1:nVol,data_filtered(:,1:10)), title('Data after Temporal Filtering')


%% Despiking the filtered-denoised ROI signals. Save the final signals in a 2D matrix named Final_ROI with 
% rows = number of volumes, columns = number of ROIs.

[ROI_final, frames] = icatb_despike_tc(data_filtered, TR); 

figure, plot(1:nVol,data(:,1:100)), title('Before Despiking')
figure, plot(1:nVol,data(:,1:100)), title('After Despiking')

%% Perform dynamic connectivity by using a phase synchrony analysis. The result will be a dFC 
% matrix, with size nxnxT, where n= the number of ROIs and T= the total number of frames.

% i)

hilbert_signal = hilbert(ROI_final');
THETA = unwrap(angle(hilbert_signal)); %phase hilbert signal

for i = 1:number_ROIs
    for j = 1:number_ROIs
        dFC(i,j,:) = cos(THETA(i,:)-THETA(j,:));
    end
end

figure, 
for i = 1:nVol
    imagesc(squeeze(dFC(:,:,i))), colorbar, colormap jet
    pause(0.5)
end


% ii)

for i = 1:nVol
    dFC(:,:,i) = zscore(squeeze(dFC(:,:,i)));
    [coeff, score] = pca(dFC(:,:,i));
    dFC_comp(:,i) = score(:,1);
end

dFC_comp = dFC_comp';


% iii)Perform a clustering analysis (hierarchical clustering with 
% inconsistency coefficient=1) on all the leading eigenvectors across time points. 
% Each centroid is a state of the brain dynamics.

Z = linkage(dFC_comp,'average');
figure, dendrogram(Z,0),title('Hierarchical Clustering on dFC_comp')

inconsistency = max(inconsistent(Z));
inconsistency = 1.1547;

T = cluster(Z,'cutoff', inconsistency);
number_clusters = max(T);
number_clusters


% iv)Perform a clustering analysis (hierarchical clustering with
% inconsistency coefficient=1) on all the leading eigenvectors across time points. Each centroid is a state of the brain dynamics.

for i = 1:number_clusters
    idx = find(T==i);
       if length(idx) == 1
        centroid(i,:) = dFC_comp(idx,:);
       else
       centroid(i,:) = nanmean(dFC_comp(idx,:));
    end
end


%% Relationship between dynamic functional features and tissues receptor/metabolic properties:

GABAa = load_untouch_nii('MNI_GABAa_flumazenil.nii');
GABAa = double(GABAa.img);

HDAC = load_untouch_nii('MNI_HDAC_Martinostat.nii');
HDAC = double(HDAC.img);

Ki_FDG = load_untouch_nii('MNI_Ki_FDG.nii');
Ki_FDG = double(Ki_FDG.img);

mGluR5_ABP = load_untouch_nii('MNI_mGluR5_ABP.nii');
mGluR5_ABP = double(mGluR5_ABP.img);


% i)Extract for each ROI of the Schaefer atlas the average GABAa, HDAC, Ki and mGluR5 
% values (as for the fMRI data: exclude voxels that feature a probability of belonging to
%WM greater than 95%). Expected matrices: GABAa_ROI, HDAC_ROI, Ki_ROI and
%mGluR5_ROI

for rr = 1 : number_ROIs
    
    ROImask = (mask==rr);

    % GABAa data exctraction
    GABAa_ROIdata = GABAa(logical(ROImask));
    GABAa_ROI(rr) = nanmean(GABAa_ROIdata);

    % HDAC data exctraction
    HDAC_ROIdata = HDAC(logical(ROImask));
    HDAC_ROI(rr) = nanmean(HDAC_ROIdata);

    % Ki data exctraction
    Ki_ROIdata = Ki_FDG(logical(ROImask));
    Ki_ROI(rr) = nanmean(Ki_ROIdata);

    % mGluR5 data exctraction
    mGluR5_ROIdata = mGluR5_ABP(logical(ROImask));
    mGluR5_ROI(rr) = nanmean(mGluR5_ROIdata);

end

% For each centroid calculated at 10.iv point, perform the fit of the
% following linear model

%iii. Save the p estimates in a matrix ESTIMATES (expected size 5 x number of clusters)


PETreg = [GABAa_ROI; HDAC_ROI; Ki_ROI; mGluR5_ROI; ones(1,number_ROIs)]'; 
ESTIMATES = inv(PETreg'*PETreg)*PETreg'*centroid';
pred = PETreg * ESTIMATES;
residual = centroid'-pred;

% Identify the state that is best represented by the PET maps by
% evaluating the the R2 value and by performing residuals analysis.

centroid = centroid';

for i = 1:number_clusters
    y = corrcoef(pred(:,i),centroid(:,i));
    R2(i)  = y(1,2);
end

R2 = R2.^2;
[maxR2 idxmax] = max(R2);


figure,
plot(1:number_ROIs, centroid(:,idxmax), 'r', 1:number_ROIs, pred(:,idxmax),'b')
title('Centroids and Model Prediction')
legend('Centroids','Model Prediction')


figure,
plot(1:number_ROIs, residual(:,idxmax),'o-k', 1:number_ROIs, zeros(1,number_ROIs),'--r')
title('Residuals')






