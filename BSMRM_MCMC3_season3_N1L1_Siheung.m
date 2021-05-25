%------------------------------------------------------% Implement Bayesian Spatial Multivariate Receptor Models (BSMRM) by MCMC% BSMRM_MCMC3_season3_N1L1.m%% Written by Eun Sug Park% Last updated May 2021 %% Need the following m-files in the same folder (or in the MATLAB work folder):% BSMRM_prem.m% Gen_mvn_restricted.m% trun_mvn2.m% Summary_BSMRM.m% BSMRM_postprocessing_qq0.m --> postprocess_2021.m (5/7/2021)% BSMRM_summary_orig.m% Candidate_Models.m % (<--Candidate_Models will need to be changed for different applications.) %% The user is free to use this program as long as% appropriate acknowledgement is given.% Reference: Park, Hopke, Kim, Tan, and Spiegelman (2017), Technometrics %------------------------------------------------------%------------------Import real data for BSMSRM ---------------%T2 = readtable('Siheung_Conc.csv');T2 = readtable('Siheung_Conc_kNN_051921.csv');date = T2(:, 'date');date_all = T2(:,{'date'});%date_all{:,3} = (1:height(date_all)).';%date_all= table2timetable(date);date = table2cell(date);date = string(date);start = datenum(2019,11,16,0,0,0);fin=datenum(2020,12,06,0,0,0);dtks = start:fin; date = table2cell(date_all);date = string(date);date = datetime(date, 'Format', 'M/dd/yyyy');dtks = date;data_all=T2{:,2:31};  %<--PM25 included (col 1), date: excludedspecies_name_all=T2.Properties.VariableNames(2:31);scaleY=1; %<--Use scaled data for MCMC estimation%data= removevars(data,{'date','location','lat','lon','PM2_5','Ca2_','K', 'S','Se','Br','Ba','PM10'});sel_species=[4 3 8 7 5 6 12 9 10 14 18 19 20 22 24 25 26 27 28 31]-1 %<-----------Can be changed.%del_species=[1];  %E.g., remove PM2.5 total concentrationdata=data_all(:,sel_species);species_name=species_name_all(sel_species);PM25=data_all(:,1);data(any(isnan(data)'),:)=[]; %Remove any rows of Y with missing values (NaNs) %------------------------------------[TN,J]=size(data)s_Siheung=[37.36924538660503 126.76686553478517];s2=s_Siheungs=s2;%s_Siheung;%input('[LAT LON] for the monitoring sites (s=s2: N by 2 matrix)=');N=size(s,1);T=TN/N;w=s_Siheung%w16;%s_Siheung;%input('[LAT LON] for the locations of underlying processes (w=w9: L by 2 matrix)=');L=size(w,1);sigma_K=1.2%sigma_K%input('Standard deviation of the Gaussian kernel (sigma_K)='); %E.g., sigma_K=0.2217 was used for the Harris County data analysis in the paper. %Need to be changed for different applications.K=zeros(N,L);dist_degree=zeros(N,L);for r=1:N    for m=1:L        dist_degree(r,m)=distance(s(r,1),s(r,2),w(m,1),w(m,2));        %K(r,m)=1/(pi*2*sigma_K^2)*exp(-dist_degree(r,m)^2/(2*sigma_K^2));           K(r,m)=exp(-dist_degree(r,m)^2/(2*sigma_K^2));  %<----5/14/2021        %K: N by L matrix of kernels    endend%------------------------------------------------------------------------------%%%---------Change made in 2021: I moved this part to line 547.%Predict_A=1;%input('Predict source contributions at unmonitored site(s)? Type in 1 for yes, 0 for no.');%if Predict_A==1%    s_new=input('[LAT LON] for unmonitored site(s): s_new=');%    JJ=size(s_new,1); %    %s_new: [LAT LON] for location(s) where source contributions are%    %predicted. %    %JJ by 2 matrix, JJ: Number of unmonitored site(s) %    %For example, s_new=[29.856 -95.452], JJ=1, in Figure 4 of the paper. %endScaleX=1 %input('Scale the data by standard deviations?  Type in 1 if yes, 0 if no.')if ScaleX==1    X=data./(kron(ones(TN,1),std(data)));            stdX=std(data);else    X=data;    stdX=ones(1,J);endH_main=3000%input('Total number of main MCMC iterations='); %Key in the run length for the main MCMC runs to achieve convergence.%E.g., 1000, 2000, 5000, 10000, 20000, etc. %Will vary for different applications.mstep_main=1;	%subsampling lag: can be changed, e.g., to 10, 20.Bi_main=0;	%Burn-in for main MCMC runsmcsize_main=(H_main-Bi_main)/mstep_main; %Size of the main MCMC samplesmcsize=mcsize_main;%-------------%Check convergence of MCMC! %Note that the inferences and summary statistics based on%MCMC samples are meaningful only if convergence has been achieved.H_prem=3000%input('Total number of preliminary MCMC iterations (including burn-in)='); %Key in the run length for the preliminary MCMC runs. %E.g., 2000, 5000, 10000, 20000, etc.Bi_prem=1000%input('Burn-in for preliminary MCMC iterations='); %E.g., 1000, 2000, 5000, 10000, etc. mstep_prem=1;	%subsampling lag: can be changedmcsize_prem=(H_prem-Bi_prem)/mstep_prem; %Size of the preliminary MCMC samples%Identifiability conditions: Prespecification of zeros in Ppreassign_zeros=1 %-----------------------------------------------%Candidate models to be compared:if preassign_zeros==1    Candidate_models_Korea_T60_finalend%Candidate models for Harris County VOC data are saved in the m-file Candidate_models.m%-----------------------------------------------qq0=12%input('qq0 (model number for which MCMC samples will be stored)=');qq_min=12%input('qq_min (first model number to be included in model comparison)=');qq_max=12%input('qq_max (last model number to be included in model comparison)='); % Values for qq0, qq_min, and qq_max can be selected from qq in Candidate_Models.mqsize=qq_max-qq_min+1;%-----------------------------------------------------------------------%Initialization for computing marginal likelihood for each modellogmD=zeros(qsize,1);  %Marginal likelihoodlogpriorstar=zeros(qsize,1); loglkd_thetastar=zeros(qsize,1); logf_thetastar=zeros(qsize,1); logh_thetastar=zeros(qsize,1);%-----------------------------------------------------------------t1=fix(clock)ticfor qq=qq_min:qq_max    %------------------------------------------------------------    %Initial value for P and A (qq: model number)    %------------------------------------------------------------    qq=qq    if preassign_zeros==1        if qq==1            Pinitial=Pinitial3_1;        elseif qq==2            Pinitial=Pinitial3_2;        elseif qq==3            Pinitial=Pinitial4_1;        elseif qq==4            Pinitial=Pinitial4_2;        elseif qq==5            Pinitial=Pinitial4_3;%P_spatial_HEI;%Pinitial2;%input('Pinitial=?')        elseif qq==6            Pinitial=Pinitial5_1;%input('Pinitial=?')        elseif qq==7            Pinitial=Pinitial5_2;%input('Pinitial=?')        elseif qq==8            Pinitial=Pinitial3_3;        elseif qq==9            Pinitial=Pinitial4_4;        elseif qq==10            Pinitial=Pinitial4_5;%P_spatial_HEI;%Pinitial2;%input('Pinitial=?')        elseif qq==11            Pinitial=Pinitial5_3;%input('Pinitial=?')        elseif qq==12            Pinitial=Pinitial5_4;%input('Pinitial=?')        end        q=size(Pinitial,1); %Estimated number of sources    else        q=qq%input('q=')        Pinitial=rand(q,J);    end        Lq=L*q;    NJ=N*J;    TL=T*L;    zeroindexall=find(Pinitial==0);    dimP=nnz(Pinitial);  %# of nonzero elements of P        %---------------------------------------------    % Prior distributions    %---------------------------------------------    % Need to be changed to appropriate distributions    % for different applications    %----------------------------------------------        %---------------------------------------------    %prior for P: vecP_0~Truncated Normal(c0_0,C0_0)    %---------------------------------------------    con_c0_P=0.5;      c0=reshape(con_c0_P*ones(q,J),q*J,1);     c0(zeroindexall)=0;    mu0_P=reshape(c0,q,J);  %matrix form of c0 <<<<<<<<<<<<<<<<<<    con_cov_P0=100;    C0=con_cov_P0*eye(q*J);     iC0=inv(C0);    c0_0=c0;    c0_0(zeroindexall)=[];	%mean of normal prior of vecP_0    C0_0=C0;    C0_0(zeroindexall,:)=[];    C0_0(:,zeroindexall)=[];%variance of normal prior of vecP_0    iC0_0=inv(C0_0);    %---------------------------------------------        %---------------------------------------------    %Prior for Sigma: inverse gamma (a0, b0)    %(parameterization: mean a0/b0)    %---------------------------------------------    a0=0.01;    b0=0.01*ones(J,1);        %---------------------------------------------    %prior for Gt: G_t~Normal(ksiG,kron(eye(L,Omega0))    %---------------------------------------------    %%ksi0=zeros(q,1);    ksi0=ones(q,1);    ksiG=kron(ones(L,1),ksi0);    Omega0=eye(q);    iOmega0=inv(Omega0);           %---------------------------------------------    %Preliminary MCMC run    %-----------------------------------------    %The purpose of this preliminary run is to get    %theta* =(P*,G*,Sigma*)    %(theta* is denoted as theta^c in the paper.)    Star_mode=1; %To take the posterior mode as theta* from the prelimanary run.    %Start_mode=0 to take the posterior mean as theta*     %from the prelimanary run.    %---------------------------------------------    %Starting values for P, Sigma for prelimenary run    P=Pinitial;    Sigma_initial=eye(J);    Sigma=Sigma_initial;    dsigma_e=diag(Sigma);  %diagonal elements of Sigma    iSigma=inv(Sigma);        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    %---------------------------------------------    BSMRM_prem  %<----- Preliminary MCMC run    %----------------------------------------------    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    %---------------------------------------------    %Main MCMC run    %---------------------------------------------    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    ii=1;    logpost_max=-10000000000;        postmodeP=zeros(q,J);    postmodeA=zeros(T*N,q);    postmodeG=zeros(TL,q);    postmodeDSigma=zeros(J,1);        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    meanP=zeros(q,J);    meanG=zeros(TL,q);    meanA=zeros(T*N,q);    meanDSigma=zeros(J,1);        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    % Calculate log(marginal Density for q=q) within MCMC using (4.2) and (4.3)    % of Park, Oh, and Guttorp (2002),     % Chemometrics and Intelligent Laboratory Systems, 60, 49�67.    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        %------------------------------------------------------------------    %Calculate Log of p(theta*|Mg) where    %thetastar=(Astar,Pstar,Sigmastar,Omegastar,Mustar),    %i.e.,    %log prior densities for A, P (for nonzero elts only), Sigma,    %Omega, Mu, and kappa at thetastar    %-----------------------------------------------------------------        %--------------------------------------    %log prior density for P at Pstar    %--------------------------------------    vecPstar=reshape(Pstar,q*J,1);%vec(Pstar)    vecPstar_0=vecPstar;    vecPstar_0(zeroindexall)=[];	%vec(nonzero elements of Pstar)    logpr_P=-(1/2)*log(det(2*pi*C0_0))-.5*((vecPstar_0-c0_0)'*iC0_0*(vecPstar_0-c0_0))-sum(log(normcdf(c0_0./sqrt(diag(C0_0)))));    % sum(log(normcdf(c0_0./sqrt(diag(C0_0))))) is the log of    % normalizing constant for the multivariate truncated normal    % distribution with mean c0_0 and the covariance matrix C0_0    % when C0_0 is diagonal.    % Same as log(mvncdf(c0_0./sqrt(diag(C0_0))));        %--------------------------------------    %log prior density for G at Gstar     %--------------------------------------    %Calculate log prior densities for G at Gstar:    logpr_G=0;    V0_G=kron(eye(L),Omega0);  %Prior variance of vecG    iV0_G=kron(eye(L),iOmega0);    for t=1:T        Gt=Gstar((t-1)*L+1:(t-1)*L+L,:);        vecGt=reshape(Gt',Lq,1);        logpr_Gt=-(1/2)*log(det(2*pi*V0_G))-.5*((vecGt-ksiG)'*iV0_G*(vecGt-ksiG))-sum(log(normcdf(ksiG./sqrt(diag(V0_G)))));        %sum(log(normcdf(ksiG./sqrt(diag(V0_G))))) is the log of        %normalizing constant for the multvariate truncated normal        %distrituion with mean ksiG and the cov matrix V0_G when        %V0_G is diagonal.;        logpr_G=logpr_G+logpr_Gt;    end        %--------------------------------------    %log prior density for Sigma at Sigmastar    %--------------------------------------    logpr_Sigma_all=0;    for j=1:J        logpr_Sigma_all=logpr_Sigma_all+a0*log(b0(j))-gammaln(a0)-(a0+1)*log(dsigma_estar(j))-(b0(j)/dsigma_estar(j));    end        %--------------------------------------    %Log of prior at theta*    %Log of p(theta*|Mg)    %--------------------------------------    logpriorstar(qq-qq_min+1)=logpr_P+logpr_G+logpr_Sigma_all;        %--------------------------------------    %Log of likelihood at theta*    %Calculate Log of l(X,Y|theta*,Mg) where    %thetastar=(Astar (or Gstar),Pstar,Sigmastar)    %--------------------------------------        %vecX=reshape(X,T*N*J,1);%<-------------5/15/2021 %vec(Xmatstar): Move    %%to back    %%%vecX=reshape(X',T*N*J,1);  %<-------------5/15/2021     loglkd_thetastar(qq-qq_min+1)=-T*N/2*log(det(2*pi*Sigmastar))-.5*trace(iSigmastar*(X-Astar*Pstar)'*(X-Astar*Pstar));        %--------------------------------------    % Log of posterior kernel at thetastar    % in Eq. (4.2) of Park et al. (2002)    %--------------------------------------    logf_thetastar(qq-qq_min+1)=loglkd_thetastar(qq-qq_min+1)+logpriorstar(qq-qq_min+1);        summand1=0;    Sum_extreme_s=0;        if qq==qq0        %vecP_nonzeroS_qq0=zeros(dimP,mcsize_main);        PS_qq0=zeros(q,J,mcsize_main);        GS_qq0=zeros(TL,q,mcsize_main);        dsigma_eS_qq0=zeros(J,mcsize_main);    end        %------------------------------------------------------------    %Starting values for P, A, G, Sigma, obtained    %from the preliminary run    %------------------------------------------------------------      P=Pstar;    G=Gstar;    A=Astar;    dsigma_e=dsigma_estar;    Sigma=Sigmastar;    iSigma=inv(Sigma);        for h=1:H_main			%Start MCMC                %------------------------------------------------------------        %Full conditional posterior for Gt        %---------------------------------------------------------------------                KPT=kron(K,P'); %NJ by Lq matrix        iV_G=kron(eye(L),iOmega0)+KPT'*kron(eye(N),iSigma)*KPT;        V_G=inv(iV_G);  %Lq by Lq matrix        for t=1:T            Xt=X((t-1)*N+1:(t-1)*N+N,:);            m_G=V_G*(KPT'*kron(eye(N),iSigma)*reshape(Xt',NJ,1)+kron(eye(L),iOmega0)*ksiG); %Lq by 1 vector            %Generate vec(Gt) from Truncated Multivariate Normal with parameters m_G and V_G                      vecGt=reshape(G((t-1)*L+1:(t-1)*L+L,:)',Lq,1);            vecGt=Gen_mvn_restricted(Lq,vecGt,m_G,V_G,zeros(Lq,1),Inf*ones(Lq,1));            G_tranpose=reshape(vecGt,q,L);            Gt=G_tranpose';            G((t-1)*L+1:(t-1)*L+L,:)=Gt;            A((t-1)*N+1:(t-1)*N+N,:)=K*Gt;        end                %------------------------------------------------------------        %Full conditional posterior for P:         % Truncated Multivariate Normal(c,C)        %-----------------------------------------------        %P=GenP_free_elt(X,J,q,P,Pinitial,C0,dsigma_e,A,mu0_P);        for j=1:J            C0j=C0((q*(j-1)+1):(q*(j-1)+q),(q*(j-1)+1):(q*(j-1)+q));            Free_elt=find(Pinitial(:,j)~=0);            dimPj=length(Free_elt);            if dimPj>0                Areduced=A(:,Free_elt);                Cj=inv(inv(C0j(Free_elt,Free_elt))+1/dsigma_e(j)*Areduced'*Areduced);                cj=Cj*(inv(C0j(Free_elt,Free_elt))*mu0_P(Free_elt,j)+1/dsigma_e(j)*Areduced'*X(:,j));                P(Free_elt,j)=Gen_mvn_restricted(dimPj, P(Free_elt,j),cj,Cj,zeros(dimPj,1),Inf*ones(dimPj,1));            end        end                %---------------------        %Full conditional posterior for the elements of dsigma_e:        %Inverse Gamma (a0+0.5*T*N, b0+0.5*d_j) in my parametrization        %In Matlab's parameterization, dsigma_e~Inverse Gamma(a0+0.5*T*N,        %1./(b0+0.5*d_j))        %----------------------------------        d=diag((X-A*P)'*(X-A*P));        idsigma_e=gamrnd(0.5*T*N+a0, 1./(b0+0.5*d));        dsigma_e=1./idsigma_e;	%posterior: Inverse gamma        Sigma=diag(dsigma_e);   %J by J matrix        iSigma=diag(idsigma_e);                if h>Bi_main            if h==Bi_main+mstep_main*(ii-1)+1                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                % Calculate Log of pi_hat(theta*|X,M)                % From (4.3) of Park et al. (2002),                % pi_hat(theta*|X,M)                %   =E[P(Gstar|Pstar,Sigmastar)*P(Pstar|G,Sigmastar)*                %   P(Sigmastar|G,P)]                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                % Full conditional posterior for G_t :                %           MVN(m_G,V_G)I(G_t>0)                %where                %V_G=inv(kron(eye(L), iOmega0)+KPT'*kron(eye(N),iSigma)*KPT);                % where KPT=kron(K,P'); %NJ by Lq matrix                %m_G=V_G*(KPT'*kron(eye(N),iSigma)*vec(Xt')+kron(eye(L),iOmega0)*ksiG)                % where ksiG=kron(ones(L,1),ksi0);                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                % Calculate P(Gstar|Pstar,Sigmastar)                KPTstar=kron(K,Pstar'); %NJ by Lq matrix                iV_Gstar=kron(eye(L),iOmega0)+KPTstar'*kron(eye(N),iSigmastar)*KPTstar;  %Lq by Lq matrix                V_Gstar=inv(iV_Gstar);%Lq by Lq matrix                logcpG=0;                for t=1:T                    vecGt=reshape(G((t-1)*L+1:(t-1)*L+L,:)',Lq,1);                    vecGtstar=reshape(Gstar((t-1)*L+1:(t-1)*L+L,:)',Lq,1);                    Xt=X((t-1)*N+1:(t-1)*N+N,:);                    m_Gstar_t=V_Gstar*(KPTstar'*kron(eye(N),iSigmastar)*reshape(Xt',NJ,1)+kron(eye(L),iOmega0)*ksiG); %Lq by 1 vector                    vectorGt=vecGtstar;                    logcpGt=0;                    for iii=1:Lq                        vectorGt(iii+1:Lq)=vecGtstar(iii+1:Lq);                        vectorGt(1:iii-1)=vecGt(1:iii-1);                        condvari=1/iV_G(iii,iii);                        ss=iV_G(iii,:)*(vectorGt-m_Gstar_t)-iV_G(iii,iii)*(vectorGt(iii)-m_Gstar_t(iii));                        condmui=m_Gstar_t(iii)-ss*condvari;                        condsdi=sqrt(condvari);                        %condmui depends on vecGtstar and vecGt                        logcpGt_ii=-0.5*log(det(2*pi*condvari))-.5*((vecGtstar(iii)-condmui)^2/condvari)-log(normcdf(condmui/condsdi));                        logcpGt=logcpGt+logcpGt_ii;                    end                    logcpG=logcpG+logcpGt;                end                                %---------------------------------------------------------                %To calculate logcpP=P(Pstar|G,Sigmastar), need to use GS (or AS).                %                %Need posterior mean and covariance for non-zero (non-fixed) elements of P                %conditional on AS and Sigmastar                %----------------------------------------------------------                vecP=reshape(P,q*J,1); %vec(P)                vecP_0=vecP;                vecP_0(zeroindexall)=[];	%vec(nonzero (free) elements of P)                                iCovP_0=kron(iSigmastar,A'*A)+iC0;                iCovP_0(zeroindexall,:)=[];                iCovP_0(:,zeroindexall)=[];                CovP_0=inv(iCovP_0);                                vecX=reshape(X,T*N*J,1);%<-------------5/15/2021: Moved to here                vecmeanP=kron(iSigmastar,A')*vecX;                  vecmeanP(zeroindexall)=[];                vecmeanP_0=CovP_0*(vecmeanP+iC0_0*c0_0);                                %-----------------                [logcpP, extreme_s]=trun_mvn(dimP,CovP_0,iCovP_0,vecmeanP_0,vecPstar_0,vecP_0);                Sum_extreme_s=Sum_extreme_s+extreme_s;                                %-----------------------------------------                %To calculate logcpSigma=P(Sigmastar|G,P),                %need to use GS (AS), PS                %-----------------------------------------                             dstar=diag((X-A*P)'*(X-A*P));                logcpSigma=0;                for kj=1:J                    logcpSigma=logcpSigma+(T*N/2+a0)*log(b0(kj)+dstar(kj)/2)-gammaln(a0+T*N/2)-(T*N/2+a0+1)*log(dsigma_estar(kj))-(b0(kj)+dstar(kj)/2)/dsigma_estar(kj);                end                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                % Compute posterior means                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                meanP=((ii-1)*meanP+P)/ii;                meanG=((ii-1)*meanG+G)/ii;                meanA=((ii-1)*meanA+A)/ii;                meanDSigma=((ii-1)*meanDSigma+dsigma_e)/ii;                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                %Store MCMC sample when qq=qq0                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                              if qq==qq0                    %vecP_nonzeroS_qq0(:,ii)=vecP_0;	 %Store nonzero elements of P                    PS_qq0(:,:,ii)=P;                    GS_qq0(:,:,ii)=G;                    dsigma_eS_qq0(:,ii)=dsigma_e;	 %Store diagonal elements of Sigma                end                                %Calculate log (conditional) likelihood-------------------------------                %for computing the posterior                loglkd=-T*N/2*log(det(2*pi*Sigma))-.5*trace(iSigma*(X-A*P)'*(X-A*P));                                %-----------------------------------                %Calculate log prior densities for P (for nonzero elts                %only), G, Sigma                %-----------------------------------                %Calculate log prior density for P for nonzero (non-fixed) elts only                %This time at vecP_0 (not at vecPstar_0)                                logpr_P=-(1/2)*log(det(2*pi*C0_0))-.5*((vecP_0-c0_0)'*iC0_0*(vecP_0-c0_0))-sum(log(normcdf(c0_0./sqrt(diag(C0_0)))));                                %Calculate log prior density for G:                logpr_G=0;                V0_G=kron(eye(L),Omega0);  %Prior variance of vecG                iV0_G=kron(eye(L),iOmega0);                for t=1:T                    Gt=G((t-1)*L+1:(t-1)*L+L,:);                    vecGt=reshape(Gt',q*L,1);                    logpr_Gt=-(1/2)*log(det(2*pi*V0_G))-.5*((vecGt-ksiG)'*iV0_G*(vecGt-ksiG))-sum(log(normcdf(ksiG./sqrt(diag(V0_G)))));                    %sum(log(normcdf(ksiG./sqrt(diag(V0_G))))) is the log of                    %normalizing constant for the multvariate truncated normal                    %distrituion with mean ksiG and the cov matrix V0_G when                    %V0_G is diagonal.                    logpr_G=logpr_G+logpr_Gt;                end                                              %Calculate log prior density for Sigma                logpr_Sigma_all=0;                for j=1:J                    logpr_Sigma_all=logpr_Sigma_all+a0*log(b0(j))-gammaln(a0)-(a0+1)*log(dsigma_e(j))-(b0(j)/dsigma_e(j));                end                                logprior=logpr_P+logpr_G+logpr_Sigma_all;                logpost=loglkd+logprior;                if logpost>logpost_max                    logpost_max=logpost;                    postmodeP=P;                    postmodeG=G;                    postmodeA=A;                    postmodeDSigma=dsigma_e;                                    end                                if ii == 1                    const_overflow1=logcpG+logcpSigma+logcpP;                end                                summand1(ii)=exp(logcpG+logcpSigma+logcpP-const_overflow1);                ii=ii+1;            end %end if h=multiple of mstep        end   %end if h>Bi             end	   %End of MCMC sampling under Model qq        h_theta=sum(summand1)/mcsize_main;        logh_thetastar(qq-qq_min+1)=log(h_theta)+const_overflow1;           %----------------------------------------    %Log of marginal likelihood: logmD    %----------------------------------------    logmD(qq-qq_min+1)=logf_thetastar(qq-qq_min+1)-logh_thetastar(qq-qq_min+1);       %------------------------------------------------------    %Prediction of source contributions at unmonitored site(s)     %(without storing MCMC samples)    %------------------------------------------------------    Predict_A=1;%input('Predict source contributions at unmonitored site(s)? Type in 1 for yes, 0 for no.');    if Predict_A==1                s_Siheung=[37.36924538660503 126.76686553478517];        s_new=s_Siheung;%input('[LAT LON] for unmonitored site(s): s_new=');        JJ=1%size(s_new,1);        %s_new: [LAT LON] for location(s) where source contributions are        %predicted.        %JJ by 2 matrix, JJ: Number of unmonitored site(s)        %For example, s_new=[29.856 -95.452], JJ=1, in Figure 4 of the paper.        Knew=zeros(JJ,L); %JJ: Number of unmonitored site(s)        dist_degree_new=zeros(JJ,L);                 %------Newly added: May 2021        Phat_orig=meanP*diag(stdX);        NC_Phat_orig=sum(Phat_orig');        D_NC_Phat_orig=diag(NC_Phat_orig);        I_NC_Phat_orig=diag(1./NC_Phat_orig);        Phat_orig_nor=I_NC_Phat_orig*Phat_orig;        %------                Anew=zeros(T,q,JJ);        for jj=1:JJ            for m=1:L                dist_degree_new(jj,m)=distance(s_new(jj,1),s_new(jj,2),w(m,1),w(m,2));                %Knew(jj,m)=1/(pi*2*sigma_K^2)*exp(-dist_degree_new(jj,m)^2/(2*sigma_K^2));                Knew(jj,m)=exp(-dist_degree_new(jj,m)^2/(2*sigma_K^2));  %<-----5/14/2021            end            for t=1:T                %Anew(t,:,jj)=Knew(jj,:)*meanG((t-1)*L+1:(t-1)*L+L,:);                Anew(t,:,jj)=Knew(jj,:)*meanG((t-1)*L+1:(t-1)*L+L,:)*D_NC_Phat_orig;            end        end    end    %-----------------------------------    %Summary statistics (posterior mean and posterior mode) of MCMC samples     %(not in the original scale if the data were scaled)    %-----------------------------------    Summary_BSMRM        %Posterior summaries in the original scale can be obtained     %by running BSMRM_postprocessing_qq0.m and/or BSMRM_summary_orig.m     %after the main MCMC run.        if qq==qq0   %<------------Corrected in 2021 (i.e., qq==0-->qq==qq0)        meanP_qq0=meanP;	        meanG_qq0=meanG;        meanA_qq0=meanA;	        meanDSigma_qq0=meanDSigma;        if Predict_A==1            Anew_qq0=Anew;        end    endend  %end for qq=qq_min:qq_max%End of the main MCMC runs for comparing models qq_min through qq_maxLogMD=logmD[mD_max, qqmD]=max(logmD);qq_mD=qqmD+qq_min-1; %qq_mD: model number with the highest marginal likelihoodtoct2=fix(clock)et=etime(t2,t1)etmin=et/60ethr=etmin/60%%--------------------------------------------------------%Postprocess the stored MCMC samples (when qq=qq0) to bring them back to the original scale.%-------------------------------------------------------postprocess=1%input('Postprocess the stored samples under Model qq0 to construct posterior intervals? Type in 1 if yes, 0 if no.')if postprocess==1        %-----------------------Postprocess    PS=PS_qq0;    GS=GS_qq0;    s_Siheung=[37.36924538660503 126.76686553478517; 37.30701056452388 126.77417322322681; 37.346780 126.740031];    s_Daegu=[35.86588034847358 128.5934264814463];    s_new=s_Siheung;    if Predict_A==1        JJ=size(s_new,1);                for jj=1:JJ            for m=1:L                dist_degree_new(jj,m)=distance(s_new(jj,1),s_new(jj,2),w(m,1),w(m,2));                %Knew(jj,m)=1/(pi*2*sigma_K^2)*exp(-dist_degree_new(jj,m)^2/(2*sigma_K^2));                Knew(jj,m)=exp(-dist_degree_new(jj,m)^2/(2*sigma_K^2));  %<-----5/14/2021            end        end                K1=Knew;        jj=1%input('Type in unmonitored site # (between 1 and JJ) where source contributions are to be predicted =');        %JJ: Number of unmonitored site(s)        %unmonitored site #: any row number in matrix s_new    else        K1=K;        jj=input('Type in site # (between 1 and N) where source contributions are to be estimated =');        %N: Number of monitoring sites        %site #: any row number in matrix s    end        %Get samples of estimated/predicted source contributions at location jj.    q=size(GS,2);    AnewS=zeros(T,q,mcsize);    for hh=1:mcsize        for t=1:T            AnewS(t,:,hh)=K1(jj,:)*GS((t-1)*L+1:(t-1)*L+L,:,hh);        end    end    AS=AnewS;        PSnor=zeros(q,J,mcsize);    ASnor=zeros(T,q,mcsize);    if ScaleX==1        stdX=std(data);    else        stdX=ones(1,J);    end    for hh=1:mcsize        Phat_orig_hh=PS(:,:,hh)*diag(stdX);        NC_Phat_orig_hh=sum(Phat_orig_hh');        D_NC_Phat_orig_hh=diag(NC_Phat_orig_hh);        I_NC_Phat_orig_hh=diag(1./NC_Phat_orig_hh);        P_normalized_hh=I_NC_Phat_orig_hh*Phat_orig_hh;        PSnor(:,:,hh)=P_normalized_hh;        ASnor(:,:,hh)=(AS(:,:,hh))*D_NC_Phat_orig_hh;    end    vecPnor=reshape(PSnor,q*J,mcsize);    vmeanPnor=mean(vecPnor');    meanPnor=reshape(vmeanPnor, q,J);    meanPnor_in_percent=[(1:J)' meanPnor'*100]    vstdPnor=std(vecPnor');    stdPnor=reshape(vstdPnor, q,J);        vecAnor=reshape(ASnor,T*q,mcsize);    vmeanAnor=mean(vecAnor');    meanAnor=reshape(vmeanAnor,T,q);    vstdAnor=std(vecAnor');    stdAnor=reshape(vstdAnor,T,q);    vmedianAnor=median(vecAnor');    medianAnor=reshape(vmedianAnor,T,q);        %-------------------------------------------------    %Compute 95% posterior intervals for P and A under Model qq0    %-------------------------------------------------    note='95% Posterior Intervals'    alpha_level=0.05%input('alpha_level=')    index_L=floor(alpha_level/2*mcsize);    index_U=ceil((1-alpha_level/2)*mcsize);    sortedPnor=sort(vecPnor');    PI_P=sortedPnor([index_L index_U],:);    LPI_P95=reshape(PI_P(1,:),q,J);    UPI_P95=reshape(PI_P(2,:),q,J);    PI_width_P=UPI_P95-LPI_P95;    mean_P_PI_width=mean(PI_width_P);    LPI_P95_in_percent=LPI_P95'*100;    UPI_P95_in_percent=UPI_P95'*100;        sortedAnor=sort(vecAnor');    PI_A=sortedAnor([index_L index_U],:);    LPI_A95=reshape(PI_A(1,:),T,q);    UPI_A95=reshape(PI_A(2,:),T,q);    PI_width_A=UPI_A95-LPI_A95;    mean_A_PI_width=mean(PI_width_A);        %-------------------------------    endbarplotP(PSnor,0.05,species_name)if q>1    pcplot(X,meanP_qq0)end%----------------------------------------------------%Construct Time-seires plots of source contributions (columns of A) for Model qq0.%-----------------------------------------------------%%tsplotA(ASnor,0.05)%tsplotA(ASnor(1:6:TN,:,:),0.05)%tsplotA_date(ASnor(1:6:T,:,:),0.05,dtks)if N==1    tsplotA(ASnor,0.05)    % tsplotA_date2(ASnor(1:6:TN,:,:),0.05,dtks)    % traceplot(5,4,PSnor,ASnor(1:6:TN,:,:),1,dsigma_eS_qq0')else    tsplotA_date(ASnor,0.05,dtks)    traceplot(5,4,PSnor,ASnor,1,dsigma_eS_qq0')endmeanAhat=mean(meanAnor)sum_avgAhat=sum(mean(meanAnor))meanAhat_percent=mean(meanAnor)./sum_avgAhat*100medianAhat=median(meanAnor)sum_medAhat=sum(median(meanAnor))medianAhat_percent=median(meanAnor)./sum_medAhat*100%---------------------------------------------------------summary_orig_scale=0%input('Obtain other posterior summaries in the original scale? Type in 1 if yes, 0 if no.');if summary_orig_scale==1    BSMRM_summary_origend