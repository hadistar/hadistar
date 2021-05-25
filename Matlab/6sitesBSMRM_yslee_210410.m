
data = readtable('6sites_forBSMRM.csv');

date = data(:, 'date');
date = date(1:6:end,:);

for i=1:height(data)
    if month(data.date(i)) <= 2
        data.season(i) = 4; % 4=winter
    elseif month(data.date(i)) <= 5
        data.season(i) = 1; % 1=spring
    elseif month(data.date(i)) <= 8
        data.season(i) = 2; % 2=summer
    elseif month(data.date(i)) <= 11
        data.season(i) = 3; % 3=autumn
    else
        data.season(i) = 4; % 4=winter
    end
end

date_all = data(:,{'date','season'});
date_all{:,3} = (1:height(date_all)).';
date_all= table2timetable(date_all);


date = table2cell(date);
date = string(date);
start = datenum(2016,01,01,0,0,0);
fin=datenum(2020,01,01,0,0,0);
dtks = start:fin; 

loc =  data(1:6,{'location','lat','lon'});
s2 = table2array(loc(:,{'lat','lon'}));
data= removevars(data,{'date','location','lat','lon','PM2_5','Ca2_','K', 'S','Se','Br','Ba','PM10'});
% for J=20
data= removevars(data,{'Cr'});
data=table2array(data);
data(any(isnan(data)'),:)=[];

species_name = readtable('6sites_input_BNFA_species_name.csv','ReadVariableNames',0);
% for J=20
species_name([14],:)=[];
species_name = table2cell(species_name);
species_name = string(species_name);


sources_name = ["Secondary Nitrate"; "Secondary Sulfate"; "Mobile"; "Coal combustion"; 
                "District heating";"Industry";"Biomass burning";"Soil";"Sea salt"]


%s_new = [37.346780, 126.740031] % Siheung
s_new=table2array(readtable('s_new.csv'));
w16 = table2array(readtable('w16.csv'));

sigma_K = distance(w16(1,1),w16(1,2),w16(2,1),w16(2,2))


% Data splitting by season

data_spring = data(data(:,end)==1,:);
data_spring(:,end) = [];
date_spring = date_all(data(:,end)==1,:);

data_summer = data(data(:,end)==2,:);
data_summer(:,end) = [];
date_summer = date_all(data(:,end)==2,:);

data_autumn = data(data(:,end)==3,:);
data_autumn(:,end) = [];
date_autumn = date_all(data(:,end)==3,:);

data_winter = data(data(:,end)==4,:);
data_winter(:,end) = [];
date_winter = date_all(data(:,end)==4,:);

%data = data_winter
%date = date_winter

% for indexing by time

indexing = date_all(timerange('2019-11-01','2019-12-31'),:);
data_selected = data(table2array(indexing(:,2)),:);
dtks = indexing.date(1:height(s2):end);

BSMRM_MCMC2


% for plot

figure
for k=1:q
    subplot(q,1,k)
    bar(Phat_sum_to_100(:,k+1))
    set(gca,'xtick',[1:18],'xticklabel',species_name)
    %title('Source - '+sources_name(k))
end

%dtks = date(1:6:end,1);
%dtks = datetime(table2array(dtks));

% time series plot
tsplotA_date(ASnor,alpha_level,dtks);

%figure
%for k=1:q
%    subplot(q,1,k)
%    plot(dtks,YS(:,k),'-',dtks,LPI_A95(:,k),'r-.',dtks,UPI_A95(:,k),'r-.')
%    axis tight
%    %xlabel('Observation #')
%    ylabel('ug/m3')
%    legend('Median','95% PI','Location','North')
%    %title('Source - '+sources_name(k))
%    set(gca,'XMinorTick','on','YMinorTick','on')
%    set(gca,'XTick', dtks)
%    datetick('x', 'mmm-dd-yy')
%    
%    if k==q
%        xlabel('Time')
%    end
%end


% Check for correlation

cor = corr(Phat_sum_to_100(:,2:end))

% version 2
figure()
imagesc(cor)
colormap((jet(20)));
h = colorbar('eastoutside');
xlabel(h, 'h', 'FontSize', 14);

set(gca,'YTick',1:q);
set(gca,'XTick',1:q);
set(gca,'YTickLabel',1:q);
set(gca,'XTickLabel',1:q);
set(gca, 'Ticklength', [0 0]);
caxis([-1,1]);
grid off
box off


% Check for ACF

traceplot_ACF_A(ASnor,3,9,9)

%---------------------------------------------------------------
% Trace plots and ACF plots along with ESS of MCMC samples for 
% elements of P, A, and Sigma for monitoring convergence and autocorrelation
%---------------------------------------------------------------
nrow=3; %Number of rows for subplots per window (page)
ncol_even=4; %Number of columns for subplots per window (page).
%<--Use an even number for ncol_even. (Either 2 or 4 is the best.)
traceplot_ACF_P(PSnor,nrow,ncol_even);
k=1; %Source number for producing traceplots for A (default=1)
%<--k can be changed to any number between 1 and q.
traceplot_ACF_A(ASnor,k,nrow,ncol_even);
%traceplot_ACF_Sigma(dsigma_eS_qq0,nrow,ncol_even);

%idCond_check

idCond_check(P)

% conditon check
cond(meanP)

barplotP(PSnor,0.05,species_name)
pcplot(X,meanP_qq0)
traceplot(5,5,PS_qq0,GS_qq0,1,dsigma_eS_qq0')










%참고자료
%Phat_orig=PS(:,:,hh)*diag(stdX);
%NC_Phat_orig=sum(Phat_orig');
%D_NC_Phat_orig=diag(NC_Phat_orig);
%I_NC_Phat_orig=diag(1./NC_Phat_orig);
%P_normalized=I_NC_Phat_orig*Phat_orig;
%PSnor=P_normalized;
%ASnor=AS*D_NC_Phat_orig;

Anewnor = []

for jj=1:JJ

    Anewnor(:,:,jj)=Anew(:,:,jj)*D_NC_Phat_orig;
end




source1selection=squeeze(Anewnor(:,4,:));

csvwrite('Anew_source1.csv',source1selection');



% PMF cond check

A = readtable('A_Seoul.csv');
A = table2array(A);

P = readtable('PMFresults_Seoul.CSV');
P = table2array(P);

cond(P)

A = readtable('A_Daejeon.csv');
A = table2array(A);

P = readtable('PMFresults_Daejeon.CSV');
P = table2array(P);

cond(P)

A = readtable('A_Gwangju.csv');
A = table2array(A);

P = readtable('PMFresults_Gwangju.CSV');
P = table2array(P);

cond(P)



A = readtable('A_Ulsan.csv');
A = table2array(A);

P = readtable('PMFresults_Ulsan.CSV');
P = table2array(P);

cond(P)





