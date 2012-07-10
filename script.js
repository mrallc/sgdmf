$(function() {

    $('#title').html("sgdmf");

    (function() {
	var i = [];

	i.push("<p>");
	i.push("This is a demo of stochastic gradient descent applied to matrix factorization");
	i.push("with nuclear-norm regularization.");
	i.push("\"lambda\" penalizes model complexity (in terms of rank), so");
	i.push("smaller lambda's yield more accuracy.");
	i.push("You can drag/drop your own images here too,");
	i.push("although obviously the technique is more appropriate for sparse matrices or non-imaging purposes.");
	i.push("See <a href='http://xoba-public.s3.amazonaws.com/2a027d92d63489e5042bc1bcdcfa42b2.pdf'>Recht et al.</a>");
	i.push("and <a href='http://xoba-public.s3.amazonaws.com/752b8adf3379ea424554f7304842edbd.pdf'>Koren et al.</a> for related work.");
	i.push("Only <span class='hl'>CHROME</span> browser seems fast enough for this, and");
	i.push("if \"time per frame\" below is highlighted,");
	i.push("try increasing \"frame period\" or reducing \"iterations per frame.\"");
	i.push("</p>");
	
	i.push('<a href="https://github.com/mrallc/sgdmf"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://s3.amazonaws.com/github/ribbons/forkme_right_green_007200.png" alt="Fork me on GitHub"></a>');

	$('#main').html(i.join(" "));
	
    })();
    
    function fmt(s) {
	s += '';
	x = s.split('.');
	x1 = x[0];
	x2 = x.length > 1 ? '.' + x[1] : '';
	var r = /(\d+)(\d{3})/;
	while (r.test(x1)) {
	    x1 = x1.replace(r, '$1' + ',' + '$2');
	}
	return x1 + x2;
    };

    function uuid() {
        var x = function() {
            return (((1+Math.random())*0x10000)|0).toString(16).substring(1);
        };
        return (x()+x()+"-"+x()+"-"+x()+"-"+x()+"-"+x()+x()+x());
    }

    var db = function(x) {
	return 10 * Math.log(x) / Math.log(10);
    }
    
    var idb = function(x) {
	return Math.pow(10,x/10);
    }

    function Status() {

	var rows = [];

	var self = this;

	var add = function(title,id) {
	    rows.push("<tr><td>"+title+":</td><td class='right' id='"+id+"'></td></tr>");
	    self[id] = function(t,hl) {
		$('#'+id).html(t);

		var highlight = false;

		if (hl != undefined) {
		    highlight = hl;
		}

		if (highlight) {
		    $('#'+id).addClass("hl");
		} else {
		    $('#'+id).removeClass("hl");
		}
	    };
	};

	add("time per frame","time");
	add("frobenius norm","norm");
	add("learning rate","gamma");
	add("maximum rank","rank");

	$('#main').append("<table>"+rows.join("")+"</table>");

    };

    function Slider(name,min,max,initialValue,cb) {

	if ($('#sliders').length == 0) {
	    $('#main').append("<table id='sliders'></table>");
	}

	var step = (max-min)/300;
	var id = uuid();

	$('#sliders').append("<tr><td>"+name+":</td><td style='width:608px'><input type='range' value='"+initialValue+"' step='"+step+"' id='S"+id+"' min='"+min+"' max='"+max+"'/></td><td id='T"+id+"'></td></tr>");

	var update = function() {
	    var v = $('#S' + id).val();
	    $('#T'+id).html(cb(v));
	};

	$('#S' + id).change(update);
	
	$('#T' + id).html(cb(initialValue));
	
    };

    var luminance = function(r,g,b) {
	return Math.round(0.3 * r + 0.59 * g + 0.11 * b);
    };
    
    function Vector(w) {
	var a = new Float32Array(new ArrayBuffer(4*w));

	this.dims = [w];

	this.get = function(i) {
	    return a[i];
	};

	this.set = function(i,v) {
	    a[i] = v;
	};

	this.inc = function(i,v) {
	    a[i] += v;
	}
    };

    function Matrix(w,h) {

	var a = new Float32Array(new ArrayBuffer(4*w*h));

	this.dims = [w,h];

	this.get = function(i,j) {
	    return a[i+j*w];
	};

	this.set =  function(i,j,v) {
	    a[i+j*w] = v;
	};

	this.inc = function(i,j,v) {
	    a[i+j*w] += v;
	};

    };
    
    var imageToMatrix = function(img) {
	
	var pix = img.data;

	var w = img.width;
	var h = img.height;
	
	var mat = new Matrix(w,h);
	
	for (var i=0; i<w; i++) {
	    for (var j=0; j<h; j++) {
		var index = 4*(i+j*w);
		mat.set(i,j,luminance(pix[index],pix[index+1],pix[index+2]));
	    }
	}
	
	return mat;

    };
    
    var drawMatrix = function(ctx,img,mat) {
	
	var pix = img.data;

	var h = img.height;
	var w = img.width;

	var f = 0;

	for (var i=0; i<w; i++) {
	    for (var j=0; j<h; j++) {
		var f0 = mat.get(i,j);
		var index = 4*(i + w*j);
		pix[index] = f0;
		pix[index+1] = f0;
		pix[index+2] = f0;
		f += (f0*f0);
	    }
	}
	
	ctx.putImageData(img,0,0);
	
	return f;

    }
    
    var rand = function(n) {
	return Math.floor(Math.random() * n);
    }

    var compute = function(state) {

	var f = function() {
	    
	    var iterate = function() {
		setTimeout(f,state.period);
	    };

	    var startTime = new Date().getTime();

	    var r = state.r;

	    var original = state.original;

	    var left = state.left;
	    var right = state.right;

	    var g = state.gamma;
	    var lambda = state.lambda;

	    var lg = new Vector(r);
	    var rg = new Vector(r);

	    var h = state.h;
	    var w = state.w;

	    var la = left.data;
	    var ra = right.data;

	    for (var iter=0; iter<state.niter; iter++) {

		var i = rand(w);
		var j = rand(h);

		var pred = 0;
		for (var k=0; k<r; k++) {
		    pred += left.get(i,k) * right.get(k,j);
		}

		var error = pred - original.get(i,j);

		for (var k=0; k<r; k++) {
		    lg.set(k, error * right.get(k,j) + lambda * left.get(i,k)/h);
		    rg.set(k, error * left.get(i,k) + lambda * right.get(k,j)/w);
		}

		for (var k=0; k<r; k++) {
		    left.inc(i,k, - g * lg.get(k));
		    right.inc(k,j, - g * rg.get(k));
		}
	    }

	    var f2 = state.draw();

	    state.stats.norm(fmt((f2/1000000).toFixed(0)));

	    var endTime = new Date().getTime();
	    
	    var time = endTime-startTime;
	    
	    if (time < state.period) {
	 	state.stats.time(time + " ms");
	    } else {
	 	state.stats.time(time + " ms", true);
	    }

	    iterate();

	};

	return f;
    }
    
    var randmat = function(n,m) {
	var mat = new Matrix(n,m);
	var count=0;
	for (var i=0; i<n; i++) {
	    for (var j=0; j<m; j++) {
		mat.set(i,j,10*Math.random());
	    }
	}
	return mat;
    };

    var run = function(img) {

	return function() {

	    var h = img.height;
	    var w = img.width;
	    
	    var stats = new Status();
	    
	    var drawCanvas = function(id,w,h) {
 		$('#main').append("<canvas id='"+id+"' height='"+h+"px' width='"+w+"px'></canvas>");
	    }

	    drawCanvas('c1',w,h);
	    drawCanvas('c2',w,h);

            var c1 = $('#c1')[0].getContext("2d");
            var c2 = $('#c2')[0].getContext("2d");
            c1.drawImage(img, 0,0);
	    
	    var imageData = c1.getImageData(0,0,w,h);
	    	    
	    var rank = 10;
	    
	    var f = new Matrix(w,h);

	    state = {

		h: h,
		w: w,
		r: rank,

		stats:stats,

		gamma:0.001,
		lambda:1000,

		period:100,
		niter:10000,

		original: imageToMatrix(imageData),

		left: randmat(w,rank),
		right: randmat(rank,h),
		
		draw: function() {
		    
		    var left = this.left;
		    var right = this.right;

		    for (var i=0; i<w; i++) {
			for (var j=0; j<h; j++) {
			    var p = 0;
			    for (var k=0; k<rank; k++) {
				p += left.get(i,k) * right.get(k,j);
			    }
			    f.set(i,j,p);
			}
		    }
		    
		    return drawMatrix(c2,imageData,f);

		}
	    };

	    stats.gamma(state.gamma);
	    stats.rank(state.r);

	    new Slider('lambda',db(100),db(50000),db(state.lambda),function(v) {
		var iv = idb(v)
		state.lambda = idb(v);
		return fmt(iv.toFixed(0));
	    });

	    new Slider('iterations',db(100),db(100000),db(state.niter),function(v) {
		var iv = idb(v)
		state.niter = idb(v);
		return fmt(iv.toFixed(0)) + " per frame";
	    });

	    new Slider('frame period',db(50),db(1000),db(state.period),function(v) {
		var iv = idb(v)
		state.period = idb(v);
		return fmt(iv.toFixed(0)) + " ms";
	    });

	    setTimeout(compute(state),state.period);
	    
	    $('#main').bind('drop',function(e) {

		e.preventDefault();

		var loadImage = function(uri) {
		    var img = new Image();
		    img.onload = function() {
		  	var c1 = $('#c1')[0].getContext("2d");
			c1.drawImage(img, 0,0);
			var imageData = c1.getImageData(0,0,w,h);
			state.original = imageToMatrix(imageData);
		    };
		    img.src = uri;
		};
		
		var dt = e.originalEvent.dataTransfer;

		var uri = dt.getData("text/uri-list");
		
		if (uri == undefined) {
		    
		    var file = dt.files[0];

		    var reader = new FileReader();

		    reader.onload = function (event) {
			loadImage(event.currentTarget.result);
		    };

		    reader.readAsDataURL(file);

		} else {
		    loadImage(uri);
		}

	    });

	}
    };

    var img = new Image();
    img.onload = run(img);
    img.src = 'mit_logo.png';

});