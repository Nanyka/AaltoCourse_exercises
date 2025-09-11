classdef llike < handle
%% LLIKE is a class of procedures for doing maximum likelihood
%  This class has static methods for binary models (probit, and logit),
%  and standard errors (assymptotic and outer product of scores approximation)
    methods (Static)
		function [ ll, llgrad, llhess] = binary( input, theta)
		%PROBIT Probit function for quasi-likelihood maximization in NPL
		%   
		%   Inputs are:
		%				input   - struct with
		%							X     - the design matrix
		%							type  - specifies the type of binary model: probit (probit2) and logit
		%							fix   - (optional) contains something to be added, for example if some theta's are fixed 
		%				theta	- parameter value
		%   Outputs are:
		%               ll     	- the likelihood value at theta
		%				llgrad 	- the gradient of the likelihood function at theta
		%				llhess 	- the hessian (approximation) at theta

		index = input.X*theta;

		if isfield(input,'fixed')
			index=index+input.fixed;
		end

		if isfield(input,'type')
			if strcmp(input.type,'probit')
				f_obs=normpdf(index,0,1);
				F_obs=bsxfun(@max,bsxfun(@min,normcdf(index,0,1),1-10^(-12)),10^(-12));
		        ll=sum(input.y.*log(F_obs)+(1-input.y).*log(1-F_obs));
				if nargout > 1
					llsc=(bsxfun(@times,input.y.*((f_obs)./F_obs)-(1-input.y).*(f_obs./(1-F_obs)),input.X))';
					llgrad=sum(llsc,2);
				%	llgrad0=input.y.*((f_obs)./F_obs)-(1-input.y).*(f_obs./(1-F_obs));
				%	llgrad=input.X'*llgrad0;
					if nargout > 2
			            %llhess=bsxfun(@times,(llgrad0.*(llgrad0+index)),input.X)'*input.X;
						llhess=llsc*llsc';
			        end

				end
		    elseif strcmp(input.type,'probit2')
				aux=2*input.y-1;
				f_obs=normpdf(index,0,1);
				F_obs=bsxfun(@max,bsxfun(@min,normcdf(aux.*index,0,1),1-10^(-12)),10^(-12));
		        ll=sum(log(F_obs));
				if nargout > 1
					llsc=(bsxfun(@times,aux.*((f_obs)./F_obs),input.X))';
					llgrad=sum(llsc,2);
		%    			llgrad0=input.y.*((f_obs)./F_obs)-(1-input.y).*(f_obs./(1-F_obs));
		%				llgrad=input.X'*llgrad0;
					if nargout > 2
			%			llhess=bsxfun(@times,(llgrad0.*(llgrad0+index)),input.X)'*input.X;
						llhess=llsc*llsc';
			        end

				end
			elseif strcmp(input.type, 'logit')
				aux=-2*input.y+1;
				F_obs=aux./(1+exp(-aux.*index));
				ll=sum(log(1./(1+exp(-aux.*index))));
				llsc=(bsxfun(@times,F_obs,input.X))';
				llgrad=sum(llsc,2);
				%TODO: Future project; not important
				%F_obs=(exp(index)./(1+exp(index)));
				
				%ll=sum(input.y.*log(F_obs)+(1-input.y).*log(1-F_obs));
				%llgrad=((input.y-exp(index))'*input.X);
				%llhess=nan;
				llhess=llsc*llsc';
			else
				warning('You did not choose a supported type. Valid models are: probit, probit2, and logit.')
			end
		else
			warning('You did not choose a type. Valid models are: probit, probit2, and logit.');
		end % end of isfield...type

		end % end of binary

		function [ ll, estim,iter,conv ] = ml( fctn,  start )
		%ESTIMATE does maximum likelihood minimization
		%   Inputs are:
		%               fctn  - is a function handle for the likelihood function
		%                       expects the log-likelihood
		%               start - a starting value for the procedure
		%   Outputs are:
		%               ll    - the likelihood value in the last iteration
		%               estim - the estimates
		%               iter  - the number of steps taken
		%               conv  - boolean flag for convergence before max_iter
		%

		estim=start;
		tolerance=10^-12;
		max_iter=200;
		conv=tolerance+1;
		step=1;
		iter=0;
		ll1=-inf;
		while abs(conv)+(conv<0)*1000 > tolerance && iter<max_iter
		    [ll0,llgrad,llhess]=fctn(estim);      % Calculate likelihood derivatives and Hessian based on the outer product of scores
		    steps=step*(llhess\llgrad);                  % Take step
		       
		    if ll0 < ll1
		    	for i_step=1:10
			    	step=step/2;
		 		    steps=step*(llhess\llgrad);                  % Take step
			        [ll0,~,~]=fctn(estim+step*steps);      % Calculate likelihood derivatives and Hessian based on the outer product of scores
		 			
		    		if ll0>ll1
		    			break
		    		end
		    	end
		    	estim0=estim;
			    estim=estim+step*steps;
			    conv= transpose(llgrad) * (llhess \ llgrad);
			    ll1=ll0;
				step=1; 	
		    else
		    	estim0=estim;
			    estim=estim+step*steps;
			    conv= transpose(llgrad) * (llhess \ llgrad);
			    ll1=ll0;
			end 
		    iter=iter+1;
		end
		ll=ll1;
		end

		function [ Pnew ] = updateP(input, model, theta, P0)
 		    theta_k_bk=theta([1:6 12:end]);
		    theta_k_mc=[theta(7:8);theta(3);theta(9:end)];
            if input.alpha < 1 % does this even work for floats??
                for i_updateP=1:model.Nmarket
                    Pnew(:,i_updateP,1)=bsxfun(@times,bsxfun(@power,normcdf(model.zt(:,:,i_updateP,1)*theta_k_bk+model.eV(:,i_updateP,1)),input.alpha),bsxfun(@power,P0(:,i_updateP,1),1-input.alpha));
                    Pnew(:,i_updateP,2)=bsxfun(@times,bsxfun(@power,normcdf(model.zt(:,:,i_updateP,2)*theta_k_mc+model.eV(:,i_updateP,2)),input.alpha),bsxfun(@power,P0(:,i_updateP,2),1-input.alpha));
                end
            else
                for i_updateP=1:model.Nmarket
                    Pnew(:,i_updateP,1)=normcdf(model.zt(:,:,i_updateP,1)*theta_k_bk+model.eV(:,i_updateP,1));
                    Pnew(:,i_updateP,2)=normcdf(model.zt(:,:,i_updateP,2)*theta_k_mc+model.eV(:,i_updateP,2));
                end
            end
        end % end updateP

		function [ se ] = se( input, theta)
			if strcmp(input.type,'probit') || strcmp(input.type,'probit2')
				F_obs = normcdf(input.X*theta(:,end) + input.fixed) ;
				f_obs = normpdf(input.X*theta(:,end) + input.fixed) ;
				lamda0 = -f_obs./(1-F_obs) ;
				lamda1 = f_obs./F_obs ;
				Avarb = bsxfun(@times,lamda0.*lamda1,input.X)'*input.X;
				Avarb = inv(-Avarb) ;
				se = sqrt(diag(Avarb)) ;
			else
				warning('You did not choose a type. Valid models are: probit, probi2, and logit.');
			end % end of isfield...type
		end % end of se



    end % end of methods
end % end of class llike