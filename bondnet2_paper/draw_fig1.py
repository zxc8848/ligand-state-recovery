"""Compact 1:1 vector overview; architecture schematic, no experimental inputs."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Arc

HERE = Path(__file__).resolve().parent
NAVY, BLUE, OCHRE, OLIVE = '#1F4E79', '#2E75B6', '#B45F06', '#6B8E23'
INK = GREY = '#222222'
WARM, LINE = '#F2F0EB', '#9BA3AF'
WIDTH, HEIGHT = 5.15, 4.62
TITLE, BODY, SMALL = 7.6, 6.3, 5.5


def tint(col, amount=.93):
    rgb=np.array([int(col[i:i+2],16) for i in (1,3,5)],dtype=float)
    return '#%02X%02X%02X' % tuple(np.rint(rgb+(255-rgb)*amount).astype(int))


def overview():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':BODY,
                         'pdf.fonttype':42,'svg.fonttype':'none',
                         'mathtext.fontset':'dejavusans'})
    fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,WIDTH),ylim=(0,HEIGHT));ax.axis('off')
    labels=[]

    def txt(x,y,s,size=BODY,col=INK,**kw):
        t=ax.text(x,y,s,fontsize=size,color=col,va='center',zorder=10,**kw)
        labels.append(t);return t

    def box(x,y,w,h,fill=WARM,edge=LINE,r=.045,lw=.65,z=1):
        p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                        facecolor=fill,edgecolor=edge,linewidth=lw,zorder=z)
        ax.add_patch(p);return p

    def arr(points,col=NAVY,lw=.85,dash=False,scale=6):
        ls=(0,(2,2)) if dash else '-'
        if len(points)>2:
            ax.plot(*zip(*points[:-1]),color=col,lw=lw,ls=ls,zorder=4)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                                    mutation_scale=scale,color=col,linewidth=lw,
                                    linestyle=ls,shrinkA=0,shrinkB=0,zorder=4))

    def panel(x,y,w,h,letter,title,col=NAVY):
        box(x,y,w,h,WARM,LINE,z=0)
        box(x+.09,y+h-.245,.16,.16,col,col,r=.025,z=2)
        txt(x+.17,y+h-.165,letter,6.3,INK if col==OLIVE else 'white',ha='center',weight='bold')
        txt(x+.33,y+h-.165,title,TITLE,weight='bold')

    def chip(x,y,w,label,col,size=BODY):
        box(x,y-.095,w,.19,WARM,LINE,r=.045,z=2)
        txt(x+w/2,y,label,size,INK,ha='center',weight='bold')

    # A: observed geometry and the supplied Q are explicitly separate.
    panel(.07,3.95,5.01,.57,'a','Heavy-atom input')
    points=np.array([[-.29,-.02],[-.14,.13],[.10,.15],[.25,-.04],
                     [.04,-.18],[-.18,-.17],[.40,.13]])*.72
    points+=np.array([.56,4.12])
    for i,j in ((0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(3,6)):
        ax.plot(points[[i,j],0],points[[i,j],1],color=tint(NAVY,.5),
                lw=.55,ls=(0,(1.4,2)),zorder=1)
    for k,(x,y) in enumerate(points):
        ax.add_patch(Circle((x,y),.027,facecolor=OCHRE if k==6 else NAVY if k==2 else NAVY,
                            edgecolor='white',lw=.5,zorder=2))
    txt(1.06,4.18,r'Atom types $Z$ + coordinates $R$',BODY)
    txt(1.06,4.025,'No H coordinates, supplied bonds or template',SMALL,GREY)
    chip(3.79,4.22,1.16,r'Total charge $Q$',NAVY,6.4)
    txt(4.37,4.035,'supplied, not inferred',SMALL,GREY,ha='center')
    arr([(.53,3.93),(.53,3.79)])
    arr([(4.38,3.93),(4.38,3.79)],OCHRE)

    # B: retain the meaningful scalar/vector architecture, not a stack of boxes.
    panel(.07,2.62,3.14,1.16,'b','Frozen PaiNN encoder',BLUE)
    # Lock icon, rather than a long training-history annotation.
    ax.add_patch(Rectangle((2.94,3.58),.065,.05,facecolor=GREY,lw=0,zorder=3))
    ax.add_patch(Arc((2.973,3.633),.048,.052,theta1=0,theta2=180,
                     lw=.7,color=GREY,zorder=3))
    y_s,y_v=3.28,3.045
    txt(.28,y_s,r'$s_i$',8,INK,ha='center')
    txt(.28,y_v,r'$\vec v_i$',8,INK,ha='center')
    # Message and update form one open interaction block, repeated four times.
    box(.57,2.94,1.53,.53,tint(BLUE,.90),tint(BLUE,.53),z=2)
    ax.plot([1.30,1.30],[2.98,3.43],color=tint(BLUE,.45),lw=.6,ls=(0,(1.5,2)),zorder=3)
    for yy in (y_s,y_v):
        ax.plot([.40,2.30],[yy,yy],color=tint(BLUE,.22),lw=1.0,zorder=3)
    txt(.94,3.395,'message',SMALL,ha='center',weight='bold')
    txt(1.69,3.395,'update',SMALL,ha='center',weight='bold')
    txt(.91,3.16,r'$\sum_j$',9,INK,ha='center')
    # Vector/scalar mixing is internal to the update, not a vector pair readout.
    for start,end,rad in (((1.48,y_v+.02),(1.48,y_s-.025),-.38),
                          ((1.91,y_s-.025),(1.91,y_v+.02),-.38)):
        ax.add_patch(FancyArrowPatch(start,end,connectionstyle=f'arc3,rad={rad}',
                                    arrowstyle='-|>',mutation_scale=5,lw=.7,
                                    color=NAVY,zorder=4,shrinkA=0,shrinkB=0))
    chip(1.36,2.82,.40,r'$\times4$',BLUE,6.2)
    ax.plot([1.56,1.56],[2.915,2.94],color=tint(BLUE,.35),lw=.65,zorder=3)
    # RBF / direction input is a compact glyph under the message block.
    gx=np.linspace(.65,1.16,80)
    for mu,c in ((.76,tint(BLUE,.4)),(.91,BLUE),(1.06,tint(BLUE,.2))):
        ax.plot(gx,2.76+.065*np.exp(-((gx-mu)/.075)**2),color=c,lw=.6,zorder=3)
    arr([(.91,2.84),(.91,2.925)],BLUE,lw=.65,scale=4)
    txt(.92,2.68,r'RBF$(r_{ij})$, $\hat r_{ij}$',SMALL,GREY,ha='center')
    # Both channels feed invariant features; no direct v_i -> invariant pair label.
    arr([(2.16,y_s),(2.35,y_s)],BLUE,scale=4)
    arr([(2.16,y_v),(2.35,y_v)],BLUE,scale=4)
    box(2.36,2.95,.72,.43,'white',tint(BLUE,.52),r=.035,z=5)
    txt(2.72,3.235,'Invariant',5.8,INK,ha='center')
    txt(2.72,3.085,'readout',5.8,INK,ha='center')
    txt(2.43,2.77,'node + symmetric',SMALL,GREY,ha='center')
    txt(2.43,2.66,'pair features',SMALL,GREY,ha='center')
    arr([(3.23,3.17),(3.39,3.17)],BLUE)

    # C: two local categorical outputs, with no global-distribution claim.
    panel(3.41,2.62,1.67,1.16,'c','State head',OCHRE)
    txt(4.245,3.445,'JointNet + Q',6.4,INK,ha='center',weight='bold')
    txt(4.245,3.295,'all-MASK first pass',SMALL,GREY,ha='center')
    for k,(v,lab) in enumerate(zip((.12,.42,.27,.06,.18),('0','1','2','3','ar'))):
        x=3.56+k*.13
        ax.add_patch(Rectangle((x,2.965),.083,v*.40,
                              color=OCHRE if k==1 else tint(OCHRE,.62),lw=0,zorder=2))
        txt(x+.0415,2.90,lab,5.3,GREY,ha='center')
    rng=np.random.default_rng(7)
    values=rng.random((5,5))
    for row in range(5):
        for col in range(5):
            box(4.47+col*.081,2.965+row*.042,.065,.032,
                tint(OCHRE,.93-.72*values[row,col]),'none',r=.006,z=2)
    txt(3.87,2.76,r'$p_e(B_e)$',7,INK,ha='center')
    txt(4.66,2.76,r'$\ell_i(q,h;B)$',7,INK,ha='center')
    txt(4.245,2.66,'5 bond labels / 25 atom states',SMALL,GREY,ha='center')
    arr([(4.25,2.60),(4.25,2.45)],OCHRE)

    # D: a single bounded decoder with three concise operations.
    panel(.07,1.35,5.01,1.08,'d','Charge-constrained local re-ranking',OLIVE)
    for x in (1.74,3.40):
        ax.plot([x,x],[1.49,2.13],color=LINE,lw=.6,zorder=1)
    txt(.22,2.055,'1  Propose bonds',6.4,weight='bold')
    for k in range(3):
        x=.24+k*.43
        box(x,1.77,.35,.19,'white',LINE,r=.025,z=2)
        for i in range(3):
            ax.plot([x+.045+i*.087,x+.11+i*.087],[1.845,1.845],color=NAVY,lw=.9,zorder=3)
        ax.plot([x+.045+k*.087,x+.11+k*.087],[1.89,1.89],color=OLIVE,lw=.9,zorder=3)
    txt(.89,1.635,'10% uncertain edges; cap 8',SMALL,GREY,ha='center')
    txt(.89,1.49,'Two labels each; beam 48',SMALL,GREY,ha='center')
    arr([(1.59,1.86),(1.84,1.86)],NAVY,scale=5)

    txt(1.90,2.055,'2  Assign atom states',6.4,weight='bold')
    txt(2.57,1.86,r'$\tilde p_i=\mathrm{softmax}(\ell_i+\log\pi_i)$',7.2,INK,ha='center')
    txt(2.57,1.65,r'$\sum_iq_i=Q$',8.5,INK,ha='center')
    txt(2.57,1.49,'Neural + training-only prior',SMALL,GREY,ha='center')
    arr([(3.25,1.86),(3.51,1.86)],OLIVE,scale=5)

    txt(3.57,2.055,'3  Build, filter, rank',6.4,weight='bold')
    chip(3.66,1.855,1.22,'Constructible first',OLIVE,6.2)
    txt(4.265,1.635,'Sanitize, connected, no radicals',SMALL,GREY,ha='center')
    txt(4.265,1.49,'Then score; all fail = failure',SMALL,GREY,ha='center')
    # Trace the output from the final operation to the complete-state card.
    arr([(4.27,1.33),(4.27,1.205),(1.25,1.205),(1.25,1.09)],OLIVE,scale=5)
    txt(2.78,1.225,'Selected complete state',SMALL,GREY,ha='center',
        bbox=dict(facecolor='white',edgecolor='none',pad=1))

    # Native and partial-graph outputs have visibly distinct roles.
    box(.07,.34,2.24,.73,WARM,LINE,z=0)
    txt(1.19,.945,'Complete-state output',6.6,ha='center',weight='bold')
    txt(1.19,.775,r'$(B^\star,h^\star,q^\star)$',10,INK,ha='center')
    txt(1.19,.57,'Native molecular identity',6.7,ha='center',weight='bold')
    txt(1.19,.415,'Predicted charge and H counts retained',SMALL,GREY,ha='center')

    box(2.82,.34,2.26,.73,WARM,LINE,z=0)
    txt(3.95,.945,'Graph-only output',6.6,ha='center',weight='bold')
    txt(3.95,.775,r'$B^\prime$',10,INK,ha='center')
    txt(3.95,.57,'Strict / resonance-aware / graph-SMILES',5.9,ha='center',weight='bold')
    txt(3.95,.415,'Partial graph, not full-state equivalence',SMALL,GREY,ha='center')
    arr([(2.33,.70),(2.80,.70)],NAVY,scale=5)
    txt(2.565,.85,'Graph-only',5.4,INK,ha='center')
    txt(2.565,.55,'rewrite',5.4,INK,ha='center')
    arr([(.48,.32),(.48,.15),(.66,.15)],OLIVE,lw=.7,dash=True,scale=5)
    txt(.73,.15,'Optional: fixed-bond state enumeration',SMALL,INK,weight='bold')
    txt(3.00,.15,r'Up to 8 species under $Q$',SMALL,GREY)

    # Boundaries and baseline text-height checks at the final print size.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for t in labels:
        b=t.get_window_extent(renderer)
        if b.x0 < -1 or b.x1 > fig.bbox.width+1 or b.y0 < -1 or b.y1 > fig.bbox.height+1:
            raise RuntimeError(f'Label outside canvas: {t.get_text()}')
        if t.get_fontsize()<5.3:
            raise RuntimeError(f'Font below configured floor: {t.get_text()}')
    out=HERE/'figures/fig1_single_stage'
    out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','svg','png'):
        fig.savefig(out.with_suffix('.'+ext),dpi=500,facecolor='white')
    plt.close(fig)
    print('Wrote compact warm-gray overview at final column width (5.15 in).')


if __name__=='__main__':
    overview()
