clear; close; clc;
%%
%---------------- npl_run.m -----------------------------
%   Code for implementing:
%	npl:  NPL (Aguirregabiria 2007) (input.alpha == 1)
%	npll: NPL-lambda (Kasahara & Shimotsu) (0 < input.alpha < 1)
%
%	Game Theoretical Model:
%	Aguirregabiria (2009)
%
%	Possible statistical assumptions for \epsilon:
%	Gaussian (probit and probit2)
%
%---------------- Data ----------------------------------
%
%	Files:
%	toivanen_waterson_no_london_120809_fixed.mat
%		- contains all the variables for the estimation.
%		  all variables are double precission (fixed)
%
%---------------- Use and ownership ---------------------
%
%	Owners of intellectual property: 
%	Maria Juul Hansen & Patrick Kofod Mogensen
%
%% Begin script
%
load('toivanen_waterson_nolondon_120809_fixed.mat')
tidy = struct('err',1,'tol',10^-6,'i_npl',1,'npl_max',120);
tidy.theta_start = zeros(15,1);
theta_k = tidy.theta_start;
% Create struct for carrying design matrices (to be constrained lated),
% and information on ML estimator and tuning parameter alpha in NPL-lambda
input = struct('type','probit2','alpha',1); % mangler at tilføje i phi! probit vs logit
% Create struct for model parameters and variables (data)
model = struct('beta',0.95, 'N',2,'Xmin',0,'Xmax',14,'nobs',size(st_dataset,1),'Nmarket',max(st_dataset.DISTRIC2));
model.Xn = model.Xmax+1;
model.xspace = [kron((model.Xmin:model.Xmax)',ones(model.Xn,1)),kron(ones(model.Xn,1),(model.Xmin:model.Xmax)')];
% Create means for constructing design matrix
st_dataset.populati=st_dataset.populati/1000;
st_dataset.density=st_dataset.populati./st_dataset.DISTRIC1;
statsm  = grpstats(st_dataset,{'DISTRIC2'},{'mean'}, 'DataVars',{'populati','density','GDP_PC','AVG_RENT','ctax'});
% Make sure y is _not_ single float, but double
input.y=double([st_dataset.BK_ENTDU;st_dataset.MCD_ENTD]);
% Construct model current period variables	
model.Z=npl.Z(model,statsm);
% Indeces for filling model variables into observed markets 
ind=[st_dataset.BK_STOCK*(model.Xn)+st_dataset.MCD_STOC+1,st_dataset.DISTRIC2];
% Starting values for CCPs; note: not consistent!
P0=0.8*ones(model.Xn^2,model.Nmarket,model.N);
P1=P0;
tic
%% Start the loop
while tidy.err>tidy.tol & tidy.i_npl<=tidy.npl_max
	% Create discounted, expected profit variables and shocks
    [model.zt, model.eV]=npl.Phi_vars(model, P1);
    % Use observed states to create input for pseudo maximum likelihood
	for ii=1:2110
		z_obs(ii,:,1)=model.zt(ind(ii,1),:,ind(ii,2),1);
		z_obs(ii,:,2)=model.zt(ind(ii,1),:,ind(ii,2),2);
		e_obs(ii,:)=[model.eV(ind(ii,1),ind(ii,2),1),model.eV(ind(ii,1),ind(ii,2),2)];
	end
	% Create design matrix (e's have constant theta_e=1)
	input.X=[z_obs(:,1:6,1) zeros(size(z_obs(:,1:5,2))) z_obs(:,7:end,1);zeros(size(z_obs(:,1:2,1))) z_obs(:,3,2) zeros(size(z_obs(:,4:6,1))) z_obs(:,1:2,2) z_obs(:,4:end,2)];
	input.fixed=[e_obs(:,1);e_obs(:,2)];
	% Perform pseudo maximum likelihood
    [out.ll(tidy.i_npl), theta_k, out.iter(tidy.i_npl), out.conv(tidy.i_npl)]=llike.ml(@(theta)llike.binary(input,theta),theta_k);			%possible to do in one take
	out.theta(:,tidy.i_npl)=theta_k;
	% Update CCPs
    P0=P1;
    P1=llike.updateP(input, model, theta_k, P0);
		% Print iteration
		fprintf('NPL iteration %d, loglik: %4.6f, err: %4.6f\n',tidy.i_npl,out.ll(tidy.i_npl),tidy.err);
    % Check for 'convergence'
    if tidy.i_npl>1
        tidy.err=max(abs(out.theta(:,end)-out.theta(:,end-1)));
    end
    % Increment counter
	tidy.i_npl=tidy.i_npl+1;
end
%
% Calculate standard errors
out.se=llike.se(input,out.theta(:,end));
%% End script
t=toc;
fprintf('NPL estimation complete in %4.2f seconds \n',t);