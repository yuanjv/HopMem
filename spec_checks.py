"""Numpy (float64) re-derivation of the HopMem v2.0.4 identities, bounds and worked numbers, plus the
cross-talk-floor / stability check that contradicts spec 6.4. Run: python spec_checks.py (needs numpy, scipy)."""

# ===== part 1: identities, bounds, worked numbers =====
import numpy as np
rng = np.random.default_rng(0)
def unit(x): return x/np.linalg.norm(x,axis=-1,keepdims=True)
ok = lambda name, cond, extra="": print(("PASS " if cond else "FAIL ")+name+("  "+extra if extra else ""))

# ---------- 1. invariants + decomposition under random op sequence ----------
d=12; J=np.eye(d)/d; Gam=0.3; om=0.7; eb=0.2
W=J.copy(); M=np.zeros((d,d)); S=0.0
worst=0
for k in range(200):
    dt=rng.exponential(1.0); p=1-np.exp(-Gam*dt)
    W=(1-p)*W+p*J; M*=np.exp(-Gam*dt); S*=np.exp(-Gam*dt)
    s=rng.exponential(1.0); a=-np.expm1(-om*s); eps=eb*(1-a); z=unit(rng.normal(size=d))
    T=(1-eps)*np.outer(z,z)+eps*J
    W=(1-a)*W+a*T; M=(1-a)*M+a*(1-eps)*np.outer(z,z); S=(1-a)*S+a*(1-eps)
    if rng.random()<0.3:   # reinforcement
        u=unit(rng.normal(size=d)); mu=u@W@u; ar=-np.expm1(-om*rng.exponential()); tau=1/d+(1-1/d)*ar; C=rng.random(); bb=0.5
        beta=C*min(bb,(tau-mu)/(1-mu)) if mu<tau else 0.0
        W=(1-beta)*W+beta*np.outer(u,u); M=(1-beta)*M+beta*np.outer(u,u); S=(1-beta)*S+beta
    worst=max(worst,np.abs(W-((1-S)*J+M)).max(), abs(np.trace(W)-1), abs(np.trace(M)-S))
ev=np.linalg.eigvalsh(W)
ok("1 decomposition/trace/PSD", worst<1e-12 and ev.min()>=(1-S)/d-1e-12 and ev.max()<=S+(1-S)/d+1e-12, f"worst={worst:.1e} S={S:.3f}")

# ---------- 2. commutator ----------
A=rng.normal(size=(d,d)); W0=A@A.T; W0/=np.trace(W0); a=0.3;p=0.4;z=unit(rng.normal(size=d)); T=np.outer(z,z)
wd=(1-a)*((1-p)*W0+p*J)+a*T; dw=(1-p)*((1-a)*W0+a*T)+p*J
ok("2 commutator a*p*(T-J)", np.abs((wd-dw)-a*p*(T-J)).max()<1e-14)

# ---------- 3. impure write decomposition ----------
a=0.37;eps=0.21
lhs=(1-a)*W0+a*((1-eps)*np.outer(z,z)+eps*J)
a2=a*(1-eps); pp=a*eps/(1-a+a*eps)
rhs=(1-a2)*((1-pp)*W0+pp*J)+a2*np.outer(z,z)
ok("3 impure write = diffuse(p') + pure write(a2)", np.abs(lhs-rhs).max()<1e-14)

# ---------- 4. closed-form coefficients vs streaming ----------
Tn=9; s=rng.exponential(size=Tn); dts=rng.exponential(size=Tn); t=np.cumsum(dts); tq=t[-1]+0.7
Z=unit(rng.normal(size=(Tn,d)))
M=np.zeros((d,d)); tprev=0
for i in range(Tn):
    M*=np.exp(-Gam*(t[i]-tprev)); tprev=t[i]
    a=-np.expm1(-om*s[i]); eps=eb*(1-a); M=(1-a)*M+a*(1-eps)*np.outer(Z[i],Z[i])
M*=np.exp(-Gam*(tq-tprev))
a=-np.expm1(-om*s); eps=eb*(1-a); inj=a*(1-eps)
later=np.cumsum(s[::-1])[::-1]-s
alpha=inj*np.exp(-(om*later+Gam*(tq-t)))
ok("4 closed-form alpha vs streaming", np.abs(M-(Z.T*alpha)@Z).max()<1e-14)

# ---------- 5. steady states + worked numbers ----------
om_=0.5;G_=0.2;s_=1;dt_=1; a=-np.expm1(-om_*s_)
Spost=a/(1-(1-a)*np.exp(-G_*dt_)); Spre=np.exp(-G_*dt_)*Spost
S=0
for _ in range(500): S=np.exp(-G_*dt_)*S; S=(1-a)*S+a
ok("5 steady state formulas", abs(S-Spost)<1e-12 and abs(Spost-0.78160)<1e-5 and abs(Spre-0.63992)<1e-5 and abs(Spost*np.exp(-.05)-0.7435)<1e-4 and abs(Spost*np.exp(-.5)-0.4741)<1e-4, f"post={Spost:.5f} pre={Spre:.5f}")

# ---------- 6. W-form == M-form at eta' ----------
Zs=unit(rng.normal(size=(5,d))); al=np.array([.05,.1,.02,.03,.08]); M=(Zs.T*al)@Zs; S=al.sum(); W=(1-S)*J+M
q=unit(rng.normal(size=d)); eta=3.0; gam=0.02/eta
def run(Mat,eta,gam,N=20):
    u=q.copy(); out=[u.copy()]
    for _ in range(N):
        v=u+eta*(Mat@u+gam*q); u=v/np.linalg.norm(v); out.append(u.copy())
    return np.array(out)
etap=eta/(1+eta*(1-S)/d)
ok("6 W-form == M-form(eta') trajectory", np.abs(run(W,eta,gam)-run(M,etap,gam)).max()<1e-13)

# ---------- 7. energy descent for huge eta + square-summability ----------
bad=0; worst_ratio=0
for _ in range(2000):
    A=rng.normal(size=(d,rng.integers(1,d))); Wr=A@A.T; Wr/=np.trace(Wr)
    eta=10**rng.uniform(-2,4); chi=rng.uniform(0,0.99); gam=chi/eta; q=unit(rng.normal(size=d))
    u=q.copy(); E=lambda u:-0.5*u@Wr@u-gam*u@q; Es=[E(u)]; ss=0
    for _ in range(20):
        v=u+eta*(Wr@u+gam*q); un=v/np.linalg.norm(v); ss+=np.sum((un-u)**2); u=un; Es.append(E(u))
    if np.any(np.diff(Es)>1e-12): bad+=1
    bound=2*eta*(Es[0]-Es[-1])/(2-chi); worst_ratio=max(worst_ratio, ss/bound if bound>0 else 0)
ok("7 energy descent (2000 random cases, eta up to 1e4)", bad==0, f"violations={bad}")
ok("7b sum ||du||^2 <= 2eta(E0-EN)/(2-chi)", worst_ratio<=1+1e-9, f"max ratio={worst_ratio:.4f}")

# ---------- 8. calibration formula (single trace, d=384) ----------
d=384; rows=[(0.1,0.005,0.05),(1,1e-9,0.05),(1,0.01,0.05),(1,0.01,0.01),(10,0.05,0.01)]
print("   calibration table (nonlinear sim vs formula vs spec):")
spec=[0.09,0.62,0.57,0.16,0.63]
for (eta,chi,m),sp in zip(rows,spec):
    z=np.zeros(d); z[0]=1; phi=0.1; q=np.zeros(d); q[0]=np.cos(phi); q[1]=np.sin(phi)
    gam=chi/eta; u=q.copy()
    for _ in range(20):
        v=u+eta*(m*z*(z@u)+gam*q); u=v/np.linalg.norm(v)
    psi=phi-np.arccos(np.clip(u@z,-1,1)); g=eta*m
    form=g/(g+chi)*(1-(1+g+chi)**-20)
    print(f"     eta={eta:<5} chi={chi:<7g} m={m:<5} sim={psi/phi:.3f} formula={form:.3f} spec={sp}")
    assert abs(psi/phi-form)<0.01 and abs(form-sp)<0.01

# ---------- 9. cross-talk floor (Arm R) ----------
d=384;Tn=20;al=0.05;gam=0.01;eta=1;N=20
xs=[];res={0.1:[],0.6:[]}; pred=[]
for trial in range(200):
    Z=unit(rng.normal(size=(Tn,d))); zt=Z[0]
    x=sum(al*(Z[j]@zt)*(Z[j]-(Z[j]@zt)*zt) for j in range(1,Tn)); xs.append(np.linalg.norm(x))
    M=(Z.T*al)@Z
    for phi in res:
        e=unit(rng.normal(size=d)-0); e-=(e@zt)*zt; e=unit(e); q=np.cos(phi)*zt+np.sin(phi)*e
        u=q.copy()
        for _ in range(N):
            v=u+eta*(M@u+gam*q); u=v/np.linalg.norm(v)
        res[phi].append((phi-np.arccos(np.clip(u@zt,-1,1)))/phi)
print(f"   ||x_perp|| measured {np.mean(xs):.5f}  predicted {np.sqrt((Tn-1)*al**2*(d-1)/(d*(d+2))):.5f}  (spec 0.01098 / 0.01108)")
print(f"   theta_floor pred {al*np.sqrt((Tn-1)/d)/(al+gam):.4f}")
print(f"   closed fraction phi=0.1: {np.mean(res[0.1]):+.2f} (spec -0.83)   phi=0.6: {np.mean(res[0.6]):+.2f} (spec +0.44)")

# ---------- 9b. phi=0 equilibrium angle vs floor (run to convergence) ----------
angs=[]
for trial in range(100):
    Z=unit(rng.normal(size=(Tn,d))); zt=Z[0]; M=(Z.T*al)@Z; q=zt.copy(); u=q.copy()
    for _ in range(400):
        v=u+eta*(M@u+gam*q); u=v/np.linalg.norm(v)
    angs.append(np.arccos(np.clip(u@zt,-1,1)))
print(f"   converged angle at phi=0: {np.mean(angs):.4f} vs floor {al*np.sqrt((Tn-1)/d)/(al+gam):.4f}")

# ---------- 10. reinforcement + purity change ----------
d=12;J=np.eye(d)/d
A=rng.normal(size=(d,3)); W0=0.6*J+0.4*(A@A.T)/np.trace(A@A.T)
u=unit(rng.normal(size=d)); beta=0.3; mu=u@W0@u; P=np.trace(W0@W0)
W1=(1-beta)*W0+beta*np.outer(u,u)
ok("10 purity change formula", abs(np.trace(W1@W1)-P-(2*beta*(mu-P)+beta**2*(1+P-2*mu)))<1e-14)
tau=0.4; b=(tau-mu)/(1-mu) if mu<tau else 0
if mu<tau: ok("10b reinforcement hits tau", abs(u@((1-b)*W0+b*np.outer(u,u))@u-tau)<1e-14)

# ---------- 11. relative spectral invariance & purity formula ----------
M=(Z.T*al)@Z if False else None
d=10;J=np.eye(d)/d; Zs=unit(rng.normal(size=(4,d))); alp=np.array([.1,.2,.05,.15]); M=(Zs.T*alp)@Zs; S=alp.sum(); W=(1-S)*J+M
lam=np.linalg.eigvalsh(W)[::-1]; r0=(lam[0]-lam[1])/(lam[0]-1/d)
f=np.exp(-0.7); W2=f*W+(1-f)*J; lam2=np.linalg.eigvalsh(W2)[::-1]; r1=(lam2[0]-lam2[1])/(lam2[0]-1/d)
ok("11 normalized gap invariant under diffusion", abs(r0-r1)<1e-13)
ok("11b P=(1-S^2)/d+||M||_F^2", abs(np.trace(W@W)-((1-S**2)/d+np.sum(M*M)))<1e-13)

# ---------- 12. closure at equilibrium for isolated trace: exact planar fixed point ----------
from scipy.optimize import brentq
m0=0.05;gam=0.01;phi=0.1
th=brentq(lambda t: gam*np.sin(phi-t)-0.5*m0*np.sin(2*t),0,phi)
print(f"   equilibrium closure exact {(phi-th)/phi:.4f} vs m/(m+gamma) {m0/(m0+gam):.4f}; t_1/2cl={np.log(m0/gam)/1e-3:.1f}")

# ---------- 13. E[t^2(1-t^2)] ----------
d=384; t=unit(rng.normal(size=(200000,d)))[:,0]
print(f"   E[t^2(1-t^2)] MC {np.mean(t**2*(1-t**2)):.6e} vs {(d-1)/(d*(d+2)):.6e}")

# ---------- 14. float32 claims ----------
for x in [1e-9,1e-8,3e-8,1e-7,3e-7,1e-6]:
    naive=np.float32(1)-np.exp(np.float32(-x)); true=-np.expm1(-x)
    print(f"   x={x:.0e} naive1-exp(-x)={float(naive):.3e} true={true:.3e} relerr={abs(float(naive)-true)/true:.2f}")

# ===== part 2: is the 6.4 cross-talk floor a stable fixed point? =====
import numpy as np
rng=np.random.default_rng(1)
def unit(x): return x/np.linalg.norm(x,axis=-1,keepdims=True)
d=384; T=20; al=0.05; eta=1.0

def run(M,q,gam,N,u0=None):
    u=q.copy() if u0 is None else u0.copy()
    for _ in range(N):
        v=u+eta*(M@u+gam*q); u=v/np.linalg.norm(v)
    return u

# --- A. Hessian at z_i (cue exactly on target) and N-dependence ---
print("A) gamma=0.01 (init), T=20, equal masses 0.05, d=384, cue = target")
Z=unit(rng.normal(size=(T,d))); zt=Z[0]; M=(Z.T*al)@Z
P=np.eye(d)-np.outer(zt,zt); mu=zt@M@zt; gam=0.01
H=(mu+gam)*P-P@M@P
ev=np.linalg.eigvalsh(H)
print(f"   mu=z'Mz={mu:.4f}  mu+gamma={mu+gam:.4f}  lambda_max(P M_oth P)={np.linalg.eigvalsh(P@M@P).max():.4f}")
print(f"   tangent Hessian eigenvalue min={ev.min():+.4f}  (negative => z_i is a SADDLE of E)")
for N in [20,50,100,400,2000]:
    u=run(M,zt,gam,N); ang=np.arccos(np.clip(u@zt,-1,1))
    other=np.degrees(np.max([np.arccos(np.clip(u@Z[j],-1,1)) for j in range(1,T)]) )
    print(f"   N={N:5d}: angle to target {ang:.3f} rad, nearest other trace {np.min([np.arccos(np.clip(u@Z[j],-1,1)) for j in range(1,T)]):.3f} rad, u'Mu={u@M@u:.4f} (lam_max(M)={np.linalg.eigvalsh(M).max():.4f})")

# --- B. gamma sweep: converged angle vs spec floor vs corrected H^-1 x_perp ---
print("\nB) gamma sweep, converged (N=3000) angle at phi=0, mean over 40 draws")
print("   gamma | spec floor |  corrected |  measured | frac of draws stable (H>0)")
for gam in [0.01,0.02,0.03,0.05,0.08,0.12,0.2]:
    ms=[];sp=[];co=[];st=0
    for _ in range(40):
        Z=unit(rng.normal(size=(T,d))); zt=Z[0]; M=(Z.T*al)@Z
        P=np.eye(d)-np.outer(zt,zt); mu=zt@M@zt; x=P@M@zt
        H=(mu+gam)*P-P@M@P; Hv=np.linalg.eigvalsh(H)
        stable=Hv[1:].min()>0     # one zero eigenvalue along z itself (P is rank d-1)
        st+=stable
        sp.append(np.linalg.norm(x)/(al+gam))
        if stable:
            w=np.linalg.solve(H+np.outer(zt,zt),x); co.append(np.linalg.norm(w))
        u=run(M,zt,gam,3000); ms.append(np.arccos(np.clip(u@zt,-1,1)))
    print(f"   {gam:5.2f} | {np.mean(sp):.4f}     | {np.mean(co) if co else float('nan'):.4f}     | {np.mean(ms):.4f}    | {st/40:.2f}")

# --- C. predicted stability threshold gamma > 2 alpha sqrt((T-1)/d) ---
print(f"\nC) predicted threshold gamma_c = 2*alpha*sqrt((T-1)/d) = {2*al*np.sqrt((T-1)/d):.4f}")
for Tn in [5,10,20,50,100]:
    gc=2*al*np.sqrt((Tn-1)/d)
    # empirical: smallest gamma at which >=50% draws are stable
    def frac(gam):
        s=0
        for _ in range(30):
            Z=unit(rng.normal(size=(Tn,d))); zt=Z[0]; M=(Z.T*al)@Z
            P=np.eye(d)-np.outer(zt,zt); mu=zt@M@zt
            Hv=np.linalg.eigvalsh((mu+gam)*P-P@M@P); s+= Hv[1:].min()>0
        return s/30
    gs=np.linspace(0.2*gc,2.0*gc,25); fr=[frac(g) for g in gs]
    emp=next((g for g,f in zip(gs,fr) if f>=0.5),float('nan'))
    print(f"   T={Tn:3d}: predicted gamma_c={gc:.4f}  empirical 50%-stable gamma={emp:.4f}")

# --- D. closure vs capacity tradeoff at equilibrium ---
print("\nD) best possible converged closure at the stability edge, closure = alpha/(alpha+gamma_c)")
for Tn in [5,20,50,100,200]:
    gc=2*al*np.sqrt((Tn-1)/d); print(f"   T={Tn:3d}: gamma_c/alpha={gc/al:.3f}  max closure ~ {1/(1+gc/al):.2f}")
