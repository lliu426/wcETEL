from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.domain_types import ScaledImage
import numpy as np
from pysdot import PowerDiagram
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from numpy.linalg import cond
from scipy.stats import beta
from scipy.stats import norm
from functools import partial
import matplotlib.pyplot as plt
from IPython.display import display

#This overrides a function in pysdot.PowerDiagram in order to display a powerDiagram class object with a title
def new_display_jupyter(self, disp_centroids=True, disp_positions=True, disp_ids=True, disp_arrows=False, hide_after=-1,title = "",saveDir = ""):
        pd_list = """
            var pd_list = [];
        """.replace( "\n            ", "\n" )

        if hide_after == None:
            hide_after = 1e40

        # need to make an animation ?
        if type( self.positions )==list or type( self.weights )==list:
            ref_positions = self.positions
            ref_weights = self.weights
            for i in range( len( ref_positions ) ):
                if type( ref_positions )==list:
                    self.set_positions( ref_positions[ i ] )
                if type( ref_weights )==list:
                    self.set_weights( ref_weights[ i ] )

                inst = self._updated_grid()
                pd_list += inst.display_html_canvas(
                    np.ascontiguousarray( self.positions ),
                    np.ascontiguousarray( self.weights ),
                    self.domain._inst,
                    self.radial_func.name(),
                    hide_after
                )

            self.set_positions( ref_positions )
            self.set_weights( ref_weights )
        else:
            inst = self._updated_grid()
            pd_list += inst.display_html_canvas(
                np.ascontiguousarray( self.positions ),
                np.ascontiguousarray( self.weights ),
                self.domain._inst,
                self.radial_func.name(),
                hide_after
            )

        jsct = """
            (function() {
            // geometry
            __pd_list__

            // limits
            var min_x = pd_list[ 0 ].min_x;
            var min_y = pd_list[ 0 ].min_y;
            var max_x = pd_list[ 0 ].max_x;
            var max_y = pd_list[ 0 ].max_y;
            for( var p of pd_list ) {
                min_x = Math.min( min_x, p.min_x );
                min_y = Math.min( min_y, p.min_y );
                max_x = Math.max( max_x, p.max_x );
                max_y = Math.max( max_y, p.max_y );
            }

            // display parameters
            var disp_centroids = __disp_centroids__, disp_positions = __disp_positions__, disp_ids = __disp_ids__, disp_arrows = __disp_arrows__;
            var cr = 0.52 * Math.max( max_x - min_x, max_y - min_y );
            var cx = 0.5 * ( max_x + min_x );
            var cy = 0.5 * ( max_y + min_y );
            var orig_click_x = 0;
            var orig_click_y = 0;
            var pos_click_x = 0;
            var pos_click_y = 0;
            var cur_pd = 0;

            // canvas
            var canvas = document.createElement( "canvas" );
            canvas.style.overflow = "hidden";
            // canvas.style.width = 940;
            canvas.height = 400;
            canvas.width = 940;

            if ( typeof element != "undefined" ) {
                element.append( canvas );
            } else {
                var oa = document.querySelector( "#output-area" );
                oa.removeChild( oa.lastChild );
                oa.appendChild( canvas );
            }

            function draw() {
                var w = canvas.width, h = canvas.height;
                var m = 0.5 * Math.min( w, h );
                var s = m / cr;

                var ctx = canvas.getContext( '2d' );

                ctx.setTransform( 1, 0, 0, 1, 0, 0 );
                ctx.clearRect( 0, 0, w, h );

                ctx.font = '20px sans-serif';
                ctx.fillStyle = 'black';
                var text = plot_title;
                var textWidth = ctx.measureText(text).width;
                ctx.fillText(text, (w - textWidth) / 2, 15);

                var pd = pd_list[ cur_pd % pd_list.length ];

                if ( disp_ids || disp_centroids ) {
                    ctx.lineWidth = 1;
                    ctx.font = '16px serif';
                    ctx.strokeStyle = "#FF0000";
                    for( var i = 0; i < pd.centroids.length; ++i ) {
                        var px = ( pd.centroids[ i ][ 0 ] - cx ) * s + 0.5 * w;
                        var py = ( pd.centroids[ i ][ 1 ] - cy ) * s + 0.5 * h;
                        if ( disp_ids ) {
                            ctx.fillText( String( i ), px + 5, py );
                        }

                        if ( disp_centroids ) {
                            ctx.beginPath();
                            ctx.arc( px, py, 2, 0, 2 * Math.PI, true );
                            ctx.stroke();
                        }
                    }
                }

                ctx.translate( 0.5 * w, 0.5 * h );
                ctx.scale( s, s );
                ctx.translate( - cx, - cy );

                var c = 1.0 / s;
                ctx.lineWidth = c;
                ctx.strokeStyle = "#000000";
                ctx.stroke( pd.path_int );
                ctx.strokeStyle = "rgb(0,0,0,0.2)";
                ctx.stroke( pd.path_ext );

                ctx.strokeStyle = "#0000FF";
                if ( disp_positions ) {
                    for( var i = 0; i < pd.diracs.length; ++i ) {
                        ctx.beginPath();
                        ctx.moveTo( pd.diracs[ i ][ 0 ] - 4 * c, pd.diracs[ i ][ 1 ] );
                        ctx.lineTo( pd.diracs[ i ][ 0 ] + 4 * c, pd.diracs[ i ][ 1 ] );
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.moveTo( pd.diracs[ i ][ 0 ], pd.diracs[ i ][ 1 ] - 4 * c );
                        ctx.lineTo( pd.diracs[ i ][ 0 ], pd.diracs[ i ][ 1 ] + 4 * c );
                        ctx.stroke();
                    }
                }

                ctx.strokeStyle = "#0000FF";
                if ( disp_arrows ) {
                    for( var i = 0; i < pd.diracs.length; ++i ) {
                        ctx.beginPath();
                        ctx.moveTo( pd.centroids[ i ][ 0 ], pd.centroids[ i ][ 1 ] );
                        ctx.lineTo( pd.diracs[ i ][ 0 ], pd.diracs[ i ][ 1 ] );
                        ctx.stroke();
                    }
                }

                if ( pd_list.length > 1 && cur_pd + 1 == pd_list.length ) {
                    ctx.setTransform( 1, 0, 0, 1, 0, 0 );
                    ctx.font = '16px serif';
                    ctx.strokeStyle = "#FF0000";
                    ctx.fillText( "left click to replay the animation", 10, h - 5 );
                }
            }

            function next_frame() {
                if ( cur_pd + 1 < pd_list.length ) {
                    setTimeout( next_frame, 50 );
                    cur_pd += 1;
                    draw();
                }
            }

            canvas.addEventListener( "wheel", function( e ) {  
                if ( e.shiftKey ) {
                    var w = canvas.width, h = canvas.height;
                    var r = canvas.getBoundingClientRect();
                    var m = 0.5 * Math.min( w, h );
                    var s = m / cr;

                    var d = Math.pow( 2, ( - e.wheelDeltaY / 200.0 || e.deltaY / 5.0 ) );
                    cx -= ( e.x - r.x - 0.5 * w ) * ( d - 1 ) / s;
                    cy -= ( e.y - r.y - 0.5 * h ) * ( d - 1 ) / s;
                    cr *= d;

                    draw();
                    return false;
                }
            }, false );

            canvas.addEventListener( "mousedown", function( e ) {  
                orig_click_x = e.x;
                orig_click_y = e.y;
                pos_click_x = e.x;
                pos_click_y = e.y;
            } );

            canvas.addEventListener( "mousemove", function( e ) {  
                if ( e.buttons == 1 || e.buttons == 4 ) {
                    var w = canvas.width, h = canvas.height;
                    var m = 0.5 * Math.min( w, h );
                    var s = m / cr;

                    cx -= ( e.x - pos_click_x ) / s;
                    cy -= ( e.y - pos_click_y ) / s;
                    pos_click_x = e.x;
                    pos_click_y = e.y;

                    draw();
                }
            } );

            canvas.addEventListener( "mouseup", function( e ) {  
                if ( pd_list.length > 1 && orig_click_x === e.x && orig_click_y == e.y ) {
                    setTimeout( next_frame, 50 );
                    cur_pd = 0;
                    draw();
                }
            } );

            if ( pd_list.length > 1 ) {
                setTimeout( next_frame, 50 );
            }
            draw();
            })();
        """

        jsct = jsct.replace( "\n            ", "\n" )
        jsct = jsct.replace( "__disp_centroids__", str( 1 * disp_centroids ) )
        jsct = jsct.replace( "__disp_positions__", str( 1 * disp_positions ) )
        jsct = jsct.replace( "__disp_arrows__", str( 1 * disp_arrows ) )
        jsct = jsct.replace( "__disp_ids__", str( 1 * disp_ids ) )
        jsct = jsct.replace( "__pd_list__", pd_list )

        jsct = jsct.replace(
            "// geometry",  # or any anchor line you trust will be there
            f"var plot_title = '{title}';\n// geometry"
            )
        # jsct = jsct.replace(
        #     "// geometry",  # or any anchor line you trust will be there
        #     f"var saveDir = '{saveDir}';\n// geometry"
        #     )
        # jsct = jsct.replace(
        #     "ctx.clearRect( 0, 0, w, h );",
        #     "ctx.clearRect( 0, 0, w, h );\nctx.font = '20px sans-serif'; ctx.fillStyle = 'black'; ctx.fillText(plot_title, 10, 30);"
        # )

        import IPython
        return IPython.display.Javascript( jsct )
        
    
PowerDiagram.display_jupyter = new_display_jupyter

# constructs a uniform probability density on [0,1] x [0,1]
#ConvexPolyhedra is a density object.
def make_square(box=[0,0,1,1]):
    domain = ConvexPolyhedraAssembly()
    domain.add_box([box[0], box[1]], [box[2], box[3]])
    return domain

# constructs a square domain, to be passed to the laguerre_* functions
#Note: if you pass in a proper histogram already, division by the mean won't do anything.
#division by the mean just converts improper histgorams to probability measures
def make_image_CPA(img,box=[0,0,1,1],display=False):
    img = img / ((box[2]-box[0])*(box[3]-box[1])*np.mean(img))
    if display:
        plt.imshow(img,cmap='gray',extent=[box[0],box[2],box[1],box[3]],origin='lower')
        plt.colorbar(label='Intensity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    domain = ConvexPolyhedraAssembly()
    domain.add_img([box[0], box[1]], [box[2], box[3]], img)
    return domain

#Note: if you pass in a proper histogram already, division by the sample mean won't do anything.
#division by the  mean just converts improper histgorams to probability measures
def make_image_SI(img, box=[0, 0, 1, 1],display = False):
    img = img / ((box[2] - box[0]) * (box[3] - box[1]) * np.mean(img))
    if display:
        plt.imshow(img,cmap='gray',extent=[box[0],box[2],box[1],box[3]],origin='lower')
        plt.colorbar(label='Intensity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    return ScaledImage([box[0], box[1]], [box[2], box[3]], img)

#frequency determines the number of bins per axis.
#alpha1,beta1 : the alpha, beta params for the beta density on the x-axis
#alpha2,beta2 : along the y-axis
def make_product_beta_density(alpha1,beta1,alpha2,beta2,frequency,display=False):
    t = np.linspace(0,1,frequency)
    x,y = np.meshgrid(t,t)
    img = np.zeros(shape = (frequency-1,frequency-1))
    for i in np.arange(frequency-1):
        for j in np.arange(frequency-1):
                xLow = x[i,j]
                xHigh = x[i+1,j+1]
                yLow = y[i,j]
                yHigh = y[i+1,j+1]
                img[i,j] = (beta.cdf(xHigh,alpha1,beta1)-beta.cdf(xLow,alpha1,beta1))*(beta.cdf(yHigh,alpha2,beta2)-beta.cdf(yLow,alpha2,beta2))
    img = ((frequency-1)**2)*img
    return make_image_SI(img,display=display)

# computes the areas of the Laguerre cells intersected with the domain, and returns it as an array
# if der = True, also returns a sparse matrix representing the Hessian of the kantorovich potential function H(psi) = int psi^c + inner(g,w)
#domain : the density (a convexPolyhedra or ScaledImage)
#Y: the data. (should be in sorted order already if d=1)
#psi: initial starting position
#distr: relevant for d=1 case. Will fill in very soon
#optional_arguments: relevant for d=1 case. Will fill in very soon.
#If d=1, the data, Y, should be in sorted, increasing order already!!!
#if d=1, domain should is a 2-d array. First entry is lower, second entry is upper
#if d=1, distr is one of:
    #normal_1d
    #beta_1d
#optional_arguments: 
    #If optional_argument == normal_1d, then specify (mean,variance)
    #If optional_arguments == beta_1d, then specify (alpha,beta)
def laguerre_areas(domain, Y, psi, der=False,dimension=2,distr = None, optional_arguments = None):
    if dimension == 2:
        pd = PowerDiagram(Y, -psi, domain)
        if der:
            N = len(psi)
            mvs = pd.der_integrals_wrt_weights()
            return mvs.v_values, csr_matrix((-mvs.m_values, mvs.m_columns, mvs.m_offsets), shape=(N,N))
        else:
            return pd.integrals()
    if dimension == 1:
        #collect the split points. This is construction of the 1-dimensional power diagram
        N = len(Y)
        pd = np.zeros(N+1)
        pd[0] = domain[0]
        for i in np.arange(N-1):
            yLow = Y[i]
            yHigh = Y[i+1]
            gLow = psi[i]
            gHigh = psi[i+1]
            threshold = (gLow-gHigh+yHigh**2-yLow**2)/(2*(yHigh-yLow))
            pd[i+1] = threshold
        pd[-1] = domain[1]

        #Now we perform the integral computation and computation of matrix of partial derivs
        if distr == "normal":
            mn = optional_arguments[0]
            var = optional_arguments[1]
            sourceCDF = partial(norm.cdf,loc = mn,scale = np.sqrt(var))
            sourcePDF = partial(norm.pdf,loc = mn,scale = np.sqrt(var))
        if distr == "beta":
            alph = optional_arguments[0]
            bet = optional_arguments[1]
            sourceCDF = partial(beta.cdf,a=alph,b=bet)
            sourcePDF = partial(beta.pdf,a=alph,b=bet)
        
        #integral computation
        cdfVals = sourceCDF(pd)
        integs = [cdfVals[i]-cdfVals[i-1] for i in np.arange(1,len(cdfVals))]

        #hessian computation: first gather the off-diagonal cells
        offDiag = [sourcePDF(pd[i])/(Y[i]-Y[i-1]) for i in np.arange(1,N)]
        
        #hessian computation: now gather all entries of the matrix into row major order
        rowSums = [-(offDiag[i]+offDiag[i+1]) for i in np.arange(len(offDiag)-1)]
        mainDiagonal = [-offDiag[0]]+rowSums+[-offDiag[-1]]
        print("The length of the main diagonal is ")
        print(len(mainDiagonal))

        Hess = diags(
            diagonals = [offDiag,mainDiagonal,offDiag],
            offsets=[-1,0,1],
            shape=(N,N),
            format='csr'
        )

        if der:
            print(Hess.diagonal())
            return integs,Hess
        else:
            return integs

#domain: in 2 dimensions, this is the density (convexPolyhedra or scaledImage).
    #In 1 dimension, this is an array with two entries (lower bound on domain ,upper bound on domain)
#distr:
    #In 2 dimensions not needed. In one dimension, this is the name of the probability distribution.
    #supported: 'normal' and 'beta'
#optional_arguments:
    #In 2 dimensions not needed. In one dimension, these are the optional arguments to go with distr
    #When using distr=normal, optional arguments correspond to [mean,variance]
    #When using distr=beta, optional arguments correspond to [alpha1,beta1]
#Y: the data
#nu: The masses
#verbose: Make true to get iteration print outs
#maxerr: When the gradient norm falls below maxerr, the second order ascent stops
#maxiter: After this many iterations of ascent, stop. 
#learningRate: Prior to backtracking on a given step, this is the initial learning rate
#illConditionThresh: Backtracking is done until the hessian is sufficiently far from being non-invertible. This number is threshold on the ratio of max singular value to min singular value to make sure this is the case.
#method: currently only second order (hessian based), is implemented
#dimension: What is the dimension of the data
#maxBacktracks: Backtracking will stop after this number of iterations. If this happens, its generally bad ( and a warning will be issued)
#beta: The multiplier during backtracking
def optimal_transport(domain, Y, nu, psi0=None, verbose=False, maxerr=1e-6, maxiter=1000,learningRate = 1.0,illConditionThresh = 10e5,method="secondOrder",dimension=2,maxBacktracks = 100,beta=3/4,distr=None,optional_arguments=None):
    if psi0 is None:
        psi0 = np.zeros(len(nu))
        
    def F(psip):
        g,h = laguerre_areas(domain, Y, np.hstack((psip,0)), der=True,dimension=dimension,distr=distr,optional_arguments=optional_arguments)
        return g[0:-1], h[0:-1,0:-1]
    
    psip = psi0[0:-1] - psi0[-1]
    nup = nu[0:-1]
    g,h = F(psip)
    for it in range(maxiter):
        err = np.linalg.norm(nup - g)
        if verbose:
            print("it %d: |err| = %g" % (it, err))
            print("The condition number of the matrix is ")
            print(cond(h.toarray()))
        if err <= maxerr:
            break
        if method == "secondOrder":
            d = spsolve(h, nup - g)
        else:
            d = nup - g
        t = learningRate
        psip0 = psip.copy()
        j=0
        while True and j < maxBacktracks:
            psip = psip0 + t*d
            try:
                g,h = F(psip)
                #compute the condition number of h
                condVal = cond(h.toarray())
                #print("The condition number is "+str(condVal))
            except ValueError:
                t = beta*t
            if np.min(g) > 0 and condVal < illConditionThresh:
                break
            else:
                t = beta*t
                j=j+1
        if j == maxBacktracks:
            print("Optimal transport backtracking is stuck on an ill conditioned location")
            return ["NA","NA"]
    return np.hstack((psip,0))

#Self-explanatory.
#distr, optional_arguments: Only for 1d case.
def computeW2_squared(Y,masses,density,plotPowerDiagram = False,dimension=2,distr = None,optional_arguments=None):
    g = optimal_transport(density,Y,masses,dimension=dimension,distr=distr,optional_arguments=optional_arguments)
    if g[0] == "NA":
        print("WARNING: optimal tranport calculation became stuck on an ill conditioned location")
        return np.inf
    if dimension == 2:
        pd = PowerDiagram(Y,-g,density)
        if plotPowerDiagram:
            toDisplay = pd.display_jupyter(disp_centroids = True,disp_positions = True,disp_ids = False,disp_arrows = True,title="Power Diagram at Solution")
            display(toDisplay)
        return (np.sum(pd.second_order_moments()))


