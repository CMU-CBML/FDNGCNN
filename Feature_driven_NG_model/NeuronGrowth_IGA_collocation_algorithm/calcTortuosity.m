function [cx,cy,tortuosity] = calcTortuosity(tip,prevTip,dist,seed_x,seed_y,changeAngle)
    [lenu,lenv] = size(dist);
    [cy,cx] = expDist(prevTip(1),prevTip(2),...
        tip(1),tip(2),changeAngle);
    dist2center = sqrt((cx-(lenu/2+seed_x)).^2+...
        (cy-(lenv/2+seed_y)).^2);
    distTipExtend = sqrt((cx-tip(1))^2+(cy-tip(2))^2);
    tortuosity = (dist(tip(:,2),tip(:,1))+distTipExtend)./(dist2center);
end