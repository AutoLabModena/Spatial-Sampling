%--------------------------------------------------------------------------
% Spatial Sampling algorithm
%
% Written by Giovanni Braglia and Davide Tebaldi, 2024
% University of Modena and Reggio Emilia
% 
% tested on MATLAB R2022a
%
%--------------------------------------------------------------------------
function out = SpatialSampling(t,x,delta,gamma)
%--------------------------------------------------------------------------
%
% Spatial Sampling algorithm: given a timeseries x(t)
% return its respective parametrized curve x(s), where
% 's' is the arc-length parameter.
% 
% Parameters:
% - t: time array
% - x: positions array
% - delta: value determining the SS interval
% - gamma: tolerance w.r.t. delta
% 
% Returns:
% - out (struct) 
%    |- tn: ss-filtered time instants 
%    |- xn: ss-filtered positions
%    |- sn: phase variable associated with xn
%    |- norm_xn: norm of xn values (=delta)
    

    out.delta=delta;                                               
    % Transpose the input trajectory, if necessary
    [nrow,ncol]=size(x);                                           
    if ncol>nrow
        x=x';
    end
    
    % Initialization
    out.xn=x(1,:);  % Filtered trajectory                                                  
    out.sn=[t(1)];  % Arc-length trajectory                                                 
    out.tn=[t(1)];  % Time vector                         
    g = gamma;
    
    xcurr=x(1,:);
    j=1;
    
    while j<size(x, 1)
        
        if norm(x(j+1,:)-xcurr) > delta + g % INSERTION
        %------------------------------------------------------------------
            a = xcurr-x(j,:);
            an = norm(a);
            
            c = x(j+1,:)-x(j,:);
            cn = norm(c);
        
            if ismembertol( an, 0, 1e-3 )
                b = delta;
            else
                c_alpha = dot(a,c)/(an*cn);
                b1 = an*c_alpha + sqrt( (an*c_alpha)^2 -an^2 + delta^2 );
                b2 = an*c_alpha - sqrt( (an*c_alpha)^2 -an^2 + delta^2 );
                if b1>0
                    b=b1;
                elseif b2>0
                    b=b2;
                end
            end
                    
            tn = b*( t(j+1)-t(j) )/cn + t(j); 
            xn = x(j,:) + b*c/cn; 
            sn = out.sn(end) + delta; 
        
            out.tn = [out.tn; tn];
            out.xn = [out.xn; xn];
            out.sn = [out.sn; sn];
        
            xcurr = out.xn(end,:);
        
            continue;
            
        elseif norm(x(j+1,:)-xcurr) <= delta+g && ...
             norm(x(j+1,:)-xcurr) >= delta-g % MATCH
        %------------------------------------------------------------------
          tn = t(j+1); 
          xn = x(j+1,:); 
          sn = out.sn(end) + delta; 
        
          out.tn = [out.tn; tn];
          out.xn = [out.xn; xn];
          out.sn = [out.sn; sn];
        
          xcurr = out.xn(end,:);
        
          j=j+1;
          continue;    
        
        elseif norm(x(j+1,:)-xcurr) < delta - g % DELETION
        %------------------------------------------------------------------
          j=j+1;
          continue;
        end
    end
    
    %
    
    % Uncomment (*) if the filtered trajectory must match final position
    % out.tn = [out.tn; t(end)]; % (*)
    % out.xn = [out.xn; x(end,:)]; % (*)
    %
    
    %
    out.norm_xn = zeros(size(out.xn,1),0);
    for j=1:size(out.xn,1)-1
        out.norm_xn(j)=norm(out.xn(j+1,:)-out.xn(j,:));
    end
    
    % out.sn = [out.sn; out.sn(end)+out.norm_xn(end)]; % (*)

end
    

