
data = xlsread('pmf_result_4sites.xlsx',1); %gwangju_profile
gwangju_G = xlsread('pmf_result_4sites.xlsx',2);
data = xlsread('pmf_result_4sites.xlsx',3); %daejeon_profile
daejeon_G = xlsread('pmf_result_4sites.xlsx',4);
data = xlsread('pmf_result_4sites.xlsx',5); %seoul_profile
seoul_G = xlsread('pmf_result_4sites.xlsx',6);
data = xlsread('pmf_result_4sites.xlsx',7); %ulsan_profile
ulsan_G = xlsread('pmf_result_4sites.xls x',8);

%Calculate_scaled_profile (source sum to 100)
%scale = sum(gwangju_profile)
%scaled_profile = (gwangju_profile./scale)*100
%sum(scaled_profile)

%Calculate_scaled_profile (source sum to 100)
scale = sum(data)
scaled_profile = (data./scale)*100
sum(scaled_profile)
writematrix(scaled_profile,'scaled_profile.csv')

% average_source_contribution
gwangju_G(gwangju_G==-999) = NaN
average_ mean(gwangju_G,'omitnan')
writematrix(scaled_profile,'scaled_profile.csv')