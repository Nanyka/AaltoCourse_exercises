classdef npl < handle
%% NPL is a class of procedures for steps related to the model in NPL (AM2002/2007)
%  This class has static methods for calculating transition probabilities, profit 
%  variables, input for Phi.

    methods (Static)
		function [ uncontran, diff, uncontran_inv ] = trans(model,belief)

		uncontran=zeros(model.Xn^2,model.Xn^2,model.Nmarket);
		diff=zeros(model.Xn^2,model.Xn^2,model.Nmarket,model.N);
		contran_0=kron(eye(model.Xn),ones(model.Xn)); 
		contran_1=kron([zeros(model.Xn-1,1) eye(model.Xn-1,model.Xn-1);zeros(1,model.Xn-1) 1],ones(model.Xn)); 
		contran_2=kron(ones(model.Xn),eye(model.Xn));
		contran_3=kron(ones(model.Xn),[zeros(model.Xn-1,1) eye(model.Xn-1,model.Xn-1);zeros(1,model.Xn-1) 1]);

		for i_m=1:model.Nmarket;
			tranxmd_bk=contran_2+bsxfun(@times,contran_3-contran_2,belief(:,i_m,2));
			tranxbk_md=contran_0+bsxfun(@times,contran_1-contran_0,belief(:,i_m,1));
			
			tottran_bk_0=bsxfun(@times,contran_0,tranxmd_bk);
			tottran_bk_1=bsxfun(@times,contran_1,tranxmd_bk);
			tottran_md_0=bsxfun(@times,contran_2,tranxbk_md);
			tottran_md_1=bsxfun(@times,contran_3,tranxbk_md);

			uncontran(:,:,i_m)=tottran_bk_0+bsxfun(@times,belief(:,i_m,1),tottran_bk_1-tottran_bk_0);
			diff(:,:,i_m,1)=tottran_bk_1-tottran_bk_0;
			diff(:,:,i_m,2)=tottran_md_1-tottran_md_0;
			if nargout > 2
				uncontran_inv=zeros(model.Xn^2,model.Xn^2,model.Nmarket);
				uncontran_inv(:,:,i_m)=inv(eye(model.Xn^2)-model.beta*uncontran(:,:,i_m));
			end
		end
		end % end trans

		function [ Z ] = Z(model, statsm)
		%% Z is a loop wrapper for Zm which is profit variables per market
		Z=zeros(model.Xn^2,10,model.Nmarket,model.N,2,2);
		for i_m=1:model.Nmarket
			Stot(1,i_m)=statsm.mean_populati(i_m);
			S=Stot(1,i_m);
			mtot(i_m,:)=[statsm.mean_density(i_m) statsm.mean_GDP_PC(i_m)/1000 statsm.mean_AVG_RENT(i_m)/1000 statsm.mean_ctax(i_m)/1000];
			m=mtot(i_m,:);
			for i_player=1:2
				for i_y1=0:1
					for i_y2=0:1
						Z(:,:,i_m,i_player,i_y1+1,i_y2+1)=npl.Zn(model,S,m,i_player,i_y1,i_y2);
					end
				end
			end
		end
		end % end Z

		function [ Zn ] = Zn(model,S, m, player,y1,y2)
		%% ZN creates profit variables for market n of m
		%  
		%   Inputs are:
		%				model.xspace - grid of observed state space
		%				S     - demand shifter (market specific constant)
		%				m     - market variables (market specific constants)
		%	Outputs are:
		%               ll     	- the likelihood value at theta
		%				llgrad 	- the gradient of the likelihood function at theta
		%				llhess 	- the hessian (approximation) at theta
		Z=nan(model.Xn^2, 6+size(m,2));

		Z1=S*ones(model.Xn^2,1);
		Z2=S*(model.xspace(:,player)+y1-model.xspace(:,3-player)-y2);
		Z3=S*(model.xspace(:,player)+y1-model.xspace(:,3-player)-y2).^2;
		Z4=-(model.xspace(:,player)+y1>0);
		Z5=-(model.xspace(:,player)+y1);
		Z6=-(model.xspace(:,player)+y1).^2.*(model.xspace(:,player)+y1>0);
		Zm=bsxfun(@times,m,model.xspace(:,player)+y1);
		Zn=bsxfun(@times,(model.xspace(:,player)+y1>0),[Z1 Z2 Z3 Z4 Z5 Z6 Zm]);
		end % end of Zn

		function [ zt ] = zs(model, T, P)
		%% ZS calculates Expected current period profit variables
		% i_p is the "player" counter; it indexes each player's elements
		% i_m indexes markets
		zt=zeros(model.Xn^2,10,422,2);
		Ez=zeros(model.Xn^2,10,model.Nmarket,2,2);
		EEz=zeros(model.Xn^2,10,model.Nmarket,2);
		for i_m=1:model.Nmarket
			for i_p=1:2
				for i_y1=1:2 % redo this? 1/2
					% First we need the expected z's over the belief of other players' actions
					Ez(:,:,i_m,i_p,i_y1)=bsxfun(@times,model.Z(:,:,i_m,i_p,i_y1,1),(1-P(:,i_m,3-i_p)))+bsxfun(@times,model.Z(:,:,i_m,i_p,i_y1,2),P(:,i_m,3-i_p));
				end
				% Then we calculate the expectation of expectations over own strategy profile (if according to beliefs)
				EEz(:,:,i_m,i_p)=bsxfun(@times,Ez(:,:,i_m,i_p,1),1-P(:,i_m,i_p))+bsxfun(@times,Ez(:,:,i_m,i_p,2),P(:,i_m,i_p));
				zV=(model.beta*T.dP(:,:,i_m,i_p))*((speye(model.Xn^2)-model.beta*T.P(:,:,i_m))\EEz(:,:,i_m,i_p));
				zt(:,:,i_m,i_p)=Ez(:,:,i_m,i_p,2)-Ez(:,:,i_m,i_p,1)+zV;
			end
		end
		end % end of zs

		function [ zt, et] = Phi_vars( model, P)
		%% Phi_vars creates profit variables for Phi
		%   Calculate expected profits 
		[T.P,T.dP]=npl.trans(model,P);   
		zt=npl.zs(model,T,P);
		et=npl.et(model,T,P);
		end % end of Phi_vars

		function [ et ] = et(model,T, P)
		et=zeros(model.Xn*model.Xn,model.Nmarket,model.N);
			for i_p=1:2
				for i_m=1:model.Nmarket
					et(:,i_m,i_p)=model.beta*T.dP(:,:,i_m,i_p)*((eye(model.Xn^2)-model.beta*T.P(:,:,i_m))\normpdf(norminv(P(:,i_m,i_p))));
				end
			end
		end % end of et
		function [ zt, et, Phi ] = Phi(model, P, theta)
		[zt,et] = npl.Phi_vars(model,P);
		if nargout > 2 
			theta_bk=theta([1:6 12:end]);
			theta_mc=theta([7:8 3 9:end]);
			for iii=1:model.Nmarket
				Phi(:,iii,1)=normcdf(zt(:,:,iii,1)*theta_bk+et(:,iii,1));
				Phi(:,iii,2)=normcdf(zt(:,:,iii,2)*theta_mc+et(:,iii,2));
			end
		end
		end % end of Phi
    end % end of methods
end % end of class npl
