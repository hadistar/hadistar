G_q5 = readtable('.\PMF_forcomparison_210508\210525q5_contributions2.csv');
P_q5 = readtable('.\PMF_forcomparison_210508\210525q5_profiles3.csv');
species_name = table2cell(P_q5(:,1));
G_q5 = table2array(G_q5(:, 2:end));
P_q5 = table2array(P_q5(:, 2:end));

G_q6 = readtable('.\PMF_forcomparison_210508\210525q6_contributions2.csv');
P_q6 = readtable('.\PMF_forcomparison_210508\210525q6_profiles3.csv');
G_q6 = table2array(G_q6(:, 2:end));
P_q6 = table2array(P_q6(:, 2:end));

G_q7 = readtable('.\PMF_forcomparison_210508\210529q7_contributions2.csv');
P_q7 = readtable('.\PMF_forcomparison_210508\210529q7_profiles3.csv');
G_q7 = table2array(G_q7(:, 2:end));
P_q7 = table2array(P_q7(:, 2:end));

G_q8 = readtable('.\PMF_forcomparison_210508\210529q8_contributions2.csv');
P_q8 = readtable('.\PMF_forcomparison_210508\210529q8_profiles3.csv');
G_q8 = table2array(G_q8(:, 2:end));
P_q8 = table2array(P_q8(:, 2:end));

G_q9 = readtable('.\PMF_forcomparison_210508\210529q9_contributions2.csv');
P_q9 = readtable('.\PMF_forcomparison_210508\210529q9_profiles3.csv');
G_q9 = table2array(G_q9(:, 2:end));
P_q9 = table2array(P_q9(:, 2:end));

G_q10 = readtable('.\PMF_forcomparison_210508\210529q10_contributions2.csv');
P_q10 = readtable('.\PMF_forcomparison_210508\210529q10_profiles3.csv');
G_q10 = table2array(G_q10(:, 2:end));
P_q10 = table2array(P_q10(:, 2:end));

% For original PMF results
P_q5_orig = readtable('.\PMF_forcomparison_210508\210525q5_profiles2_(percent of species sum).csv');
P_q5_orig = table2array(P_q5_orig(2:end, 2:end));

figure
for k=1:q
    subplot(q,1,k)
    P_bargraph = P_q5_orig(:,k);
    
    bar(1:J,P_bargraph )
    set(gca,'xtick',[1:J],'xticklabel',species_name)
    end
    axis auto
    ylabel(['Source ',num2str(k)])
    if k==1
        title('Source Composition Profiles')
    elseif k==q
        xlabel('Species')
    end
    

% Rescaling for BNFA

target1 = P_q8;
target2 = G_q8;

scale = sum(target1);
scaled_profile = (target1./scale)*100;
sum(scaled_profile);
scaled_contribution = target2.*(scale/100);


[q J] = size(scaled_profile')
temp = scaled_profile';

figure
for k=1:q
    subplot(q,1,k)
    P_bargraph = temp(k,:);
    
    bar(1:J,P_bargraph )
    set(gca,'xtick',[1:J],'xticklabel',species_name)
    end
    axis auto
    ylabel(['Source ',num2str(k)])
    if k==1
        title('Source Composition Profiles')
    elseif k==q
        xlabel('Species')
    end
    
    
temp = scaled_contribution';

figure
for k=1:q
    subplot(q,1,k)
    plot(dtks,temp(k,:),'-')
    axis tight
    legend(['Source ',num2str(k)],'Location','North')   
    
    set(gca,'XMinorTick','on','YMinorTick','on')
    set(gca,'XTick', dtks)
    datetick('x', 'mmm-dd-yy')
    
    if k==1
        legend('Source 1','95% PI','Location','North')
        title('Source Contributions')
    elseif k==2
        ylabel('Mass concentration')
    elseif k==q
        xlabel('Observation #')
    end
end




