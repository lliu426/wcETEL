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