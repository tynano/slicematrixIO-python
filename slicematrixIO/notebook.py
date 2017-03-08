from IPython.core.display import HTML
from IPython.display import Javascript
import json
from uuid import uuid4
from IPython.display import DisplayObject
from IPython.display import clear_output

class GraphEngine():
	def __init__(self, sm):
		self.sm = sm
		self.graph_count = 0
		print "initializing window.graph_data"

	def drawNetworkGraph(self, network_model, color_map = "RdBuGn", graph_style = "light", graph_layout = "force", width = 1000, height = 600, charge = -100, color_axis = "closeness_centrality", label_color = "#000", label_shadow_color = "#fffff0", min_node_size = 5):
		print self.graph_count
		clear_output()
		nodes = network_model.nodes()
		links = network_model.edges()
		embedding = network_model.embedding()
		config = """
			require.config({
				paths: {
					d3: '//cdnjs.cloudflare.com/ajax/libs/d3/4.7.1/d3.min'
				}
			});
		"""
		
		init_js  = "window.current_graph_id={};".format(self.graph_count);
		
		javascript = "window.graph_data.push(JSON.parse('{}'));".format(json.dumps({"nodes":nodes, 
																			  "links":links, 
																			  "embedding": embedding.values.tolist(),
																			  "width": width,
																			  "height": height,
																			  "color_map": color_map,
																			  "graph_style": graph_style,
																			  "graph_layout": graph_layout,
																			  "charge": charge,
																			  "color_axis": color_axis,
																			  "label_color": label_color,
																			  "label_shadow_color": label_shadow_color,
																			  "min_node_size": min_node_size,
																			  "api_region": self.sm.client.region,
																			  "api_id": self.sm.client.api,
																			  "api_key": self.sm.client.api_key,
																			  "model_name": network_model.name,
																			  "model_type": network_model.type,
																			  "graph_id": self.graph_count}))
		#print javascript
		
		
		
		js_code = """
		console.log(window.current_graph_id);
		var width = window.graph_data[window.current_graph_id]['width'];
		var height = window.graph_data[window.current_graph_id]['height'];

		var label_color = window.graph_data[window.current_graph_id]['label_color'];

		var API_region = window.graph_data[window.current_graph_id]['api_region'];
		var API_id = window.graph_data[window.current_graph_id]['api_id'];
		var API_STAGE = "development";
		var API_KEY = window.graph_data[window.current_graph_id]['api_key'];

		var url = 'https://' + API_id + '.execute-api.' + API_region + '.amazonaws.com/' + API_STAGE + '/models/call';

		var model_name = window.graph_data[window.current_graph_id]['model_name'];
		var model_type = window.graph_data[window.current_graph_id]['model_type'];
		
		var min_node_size = window.graph_data[window.current_graph_id]['min_node_size'];
		
		var charge = window.graph_data[window.current_graph_id]['charge'];

		var colorAxis = window.graph_data[window.current_graph_id]['color_axis'];
		if(colorAxis == undefined){
			colorAxis = "closeness_centrality";
		}

		var colorMap = window.graph_data[window.current_graph_id]['color_map'];
		if(colorMap == undefined){
			colorMap = "RdBuGn";
		}

		var graphStyle = window.graph_data[window.current_graph_id]['graph_style'];
		// light || dark || white || ether
		if(graphStyle == undefined){
			graphStyle = "light";
		}

		var suppressLabels;
		if(suppressLabels == undefined){
			suppressLabels = true;
		}

		var graphTitle;
		if(graphTitle == undefined){
			graphTitle = "SliceMatrix-IO Network Viewer";
		}

		var graphLayout = window.graph_data[window.current_graph_id]['graph_layout'];
		if(graphLayout == undefined){
			graphLayout = "embedding";
		}

		require(['d3'], function(d3){

			element.append('<div id = "top_bar"><b><a href = "http://www.slicematrix.com" target = "_blank" style="color:#fffff0">SliceMatrix-IO</a></b> Network View</div>');
			element.append('<div class = "loadingDialog" id = "spinner_' + window.current_graph_id +'"><div class="loader"></div><br><div id = "loading_status">loading graph data...</div></div> ');
			element.append('<div class = "content_cell" id = "content_' + window.current_graph_id +'"><canvas id = "mainEvent_' + window.current_graph_id +'" width="' + width + '" height="' + height + '"></canvas></div>');

			// update background style
			if(graphStyle == "dark"){
				d3.select("#content_" + window.current_graph_id).style("background", "rgb(20,20,20)");


				$('#top_bar').css({
					"color": "rgba(255, 255, 255, 0.88)",
					"text-shadow": "5px 5px 2px rgba(120,120,120,0.2)",
					"background-color": "rgba(220,220,220, 0.4)",
					"box-shadow": "-5px 5px 5px rgba(220,220,220,0.0)"
				});
				d3.select("#loading_status").style("color", "rgb(255, 255, 255)");
			}else if(graphStyle == "white"){
				$('#content' + window.current_graph_id).css({
					"background-image": "none",
					"background-color": "#FFF",
					"background": "rgb(256,256,256)"
				});

				d3.select("#loadingDialog").style("background", "rgba(120,120,120,0.7)");
			}else if(graphStyle == "ether"){
				$('#content' + window.current_graph_id).css({
					"background-image": "none",
					"background-color": "#000",
					"background": "-webkit-radial-gradient(circle, black 15%, rgb(5,5,10), rgb(10,10,30), rgb(15,50,50))"
				});

				$('#top_bar').css({
					"color": "rgba(255, 255, 255, 0.88)",
					"text-shadow": "5px 5px 2px rgba(120,120,120,0.2)",
					"background-color": "rgba(220,220,220, 0.4)",
					"box-shadow": "-5px 5px 5px rgba(220,220,220,0.0)"
				});

				d3.select("#loading_status").style("color", "rgb(255, 255, 255)");
			}

			//
			if(graphLayout == undefined){
				graphLayout = false;
			}

			var colormap = function(maxval, midval, minval, maxcol, midcol, mincol){
			return d3.scaleLinear()
					 .domain([minval, midval, maxval])
					 .range([mincol, midcol, maxcol]);

			}

			var colormaps = {'RdBuGn':   ['rgb(20,220,120)','rgb(20,120,220)','rgb(220,20,20)'],
						 'RdGrGn':   ['rgb(0,254,122)','rgb(185,185,185)','rgb(225,26,29)'],//2
						 'PuBuXr':   ['rgb(20,120,255)','rgb(25,25,45)','rgb(255,20,120)'],//3
						 'RdBuGnXr': ['rgb(20,220,120)','rgb(20,60,120)','rgb(220,20,20)'],//4
						 'Viridis':  ['rgb(240,249,33)','rgb(30,153,138)','rgb(19,6,137)'],//5
						 'Heat':     ['rgb(110,0,0)','rgb(255,17,0)','rgb(255,255,87)'],//9
						 'Cool':     ['rgb(6,248,255)','rgb(139,116,255)','rgb(255,0,255)'],//11
						 'Greens':   ['rgb(169,196,222)','rgb(121,198,122)','rgb(48,113,60)'],//8
						 'Blues':    ['rgb(247,252,255)','rgb(95,166,209)','rgb(19,95,167)'],//7
						 'GnBu':     ['rgb(247,252,240)','rgb(124,249,125)','rgb(8,73,138)'],//6
						 'Winter':   ['rgb(0,11,249)','rgb(0,173,169)','rgb(0,255,128)']}//10

			function getMaxOfArray(numArray) {
			  return Math.max.apply(null, numArray);
			}

			function getMinOfArray(numArray) {
			  return Math.min.apply(null, numArray);
			}

			var graph = {nodes:[], links:[]};

			var curr_colormap;

			d3.select("#loading_status").html("loaded nodes...");
			var data = window.graph_data[window.current_graph_id];
			data['nodes'].forEach(function(d){
				graph['nodes'].push({id:d});
			});

			d3.select("#loading_status").html("loaded edges...");
			data['links'].forEach(function(d){
				graph['links'].push({source:d[0], target:d[1]});
			});

			//var height = 600;
			//var width  = 1200;
			var searchRadius = 3;

			//d3.select("canvas").attr("height", height).attr("width", width);

			var transform = d3.zoomIdentity;


			//var canvas = document.querySelector("canvas"),
			var canvas = document.querySelector("#mainEvent_" + window.current_graph_id),
				context = canvas.getContext("2d");
				//width = canvas.width,
				//height = canvas.height;

			var simulation = d3.forceSimulation()
				.force("link", d3.forceLink().id(function(d) { return d.id; }))
				.force("charge", d3.forceManyBody().strength(charge))
				.force("center", d3.forceCenter(width / 2, height / 2))
				.force("X", d3.forceX().x(0.2))
				.force("Y", d3.forceY().y(0.2));

			var zoom = d3.zoom();

			var k = 1.0;

			function initialize(){
			  //zoom.scaleTo(d3.select("canvas").transition().duration(3000), 0.05);
			}

			//initialize();

			var ggraph;

			var selected_node;

			var transform;

			var ticks = 0;


			var mixcolors = function(color1, color2){
				var c1 = color1.replace("rgb(", "").replace(")", "").split(",");
				var c2 = color2.replace("rgb(", "").replace(")", "").split(",");
				var r1 = Math.round((parseInt(c1[0]) + parseInt(c2[0])) / 2);
				var g1 = Math.round((parseInt(c1[1]) + parseInt(c2[1])) / 2);
				var b1 = Math.round((parseInt(c1[2]) + parseInt(c2[2])) / 2);
				if (isNaN(r1) || isNaN(g1) || isNaN(b1)){
				  return "rgba(20,120,220,0.6)";
				}else{
				  return "rgba(" + r1.toString() + "," + g1.toString() + "," + b1.toString() + ",0.6)";
				}
			}

			var drawGraph = function(){
				console.log(graph);

				function dragsubject() {
				  var i,
					  x = transform.invertX(d3.event.x),
					  y = transform.invertY(d3.event.y),
					  dx,
					  dy;
				}


				function dragstarted() {
				  if (!d3.event.active) simulation.alphaTarget(0.3).restart();
				  d3.event.subject.fx = d3.event.subject.x;
				  d3.event.subject.fy = d3.event.subject.y;
				}

				function dragged() {
				  //d3.event.subject[0] = transform.invertX(d3.event.x);
				  //d3.event.subject[1] = transform.invertY(d3.event.y);
				  d3.event.subject.fx = d3.event.x;
				  d3.event.subject.fy = d3.event.y;
				}

				function dragended() {
				  if (!d3.event.active) simulation.alphaTarget(0);
				  d3.event.subject.fx = null;
				  d3.event.subject.fy = null;
				}

				function drawLink(d) {
				  context.moveTo(d.source.x, d.source.y);
				  context.lineTo(d.target.x, d.target.y);
				}

				function drawNode(d) {
				  context.moveTo(d.x + get_current_node_size(), d.y);
				  context.arc(d.x, d.y, get_current_node_size(), 0, 2 * Math.PI);
				  //context.fill();
				}

				function drawLabel(d, id) {
				  //console.log(d);
				  context.shadowBlur = 10;
				  context.shadowColor = "rgba(120,120,120,0.3)";
				  context.fillText(d.id, d.x,d.y);
				  //context.fill();
				}

				function rgb2a(rgb_string, alpha){
					//console.log(rgb_string);
				  rgb_string = rgb_string.replace(")", "," + alpha + ")").replace("rgb", "rgba");
				  //rgb_string = "rgba(0,0,0,0.3)";
				  return rgb_string 
				}

				function get_current_alpha(){
				  return 1.0 - Math.min(transform.k, 0.5);
				}

				function get_current_font_size(){
				  return 10 / Math.pow(transform.k, 0.5);
				}

				function get_current_highlighted_font_size(){
				  return Math.min(720 * transform.k, 20);
				}

				function get_current_link_width(){
				  return Math.max(1.5, 0.5/ transform.k);
				}

				function get_current_node_size(){
				  return Math.min(60 * transform.k, min_node_size);
				}

				function get_current_deg_thresh(){
				  if(k >= 0.4){
					return 1;
				  }else{
					return 2;
				  }
				}


				ggraph = graph;
				  setTimeout(initialize, 0);

				  if(graphLayout == "force"){

					  simulation
						  .nodes(graph.nodes)
						  .on("tick", ticked);

					  simulation.force("link")
						  .links(graph.links);

				  }else{
						  simulation
						  .nodes(graph.nodes);
						  //.on("tick", ticked);

					  simulation.force("link")
						  .links(graph.links);
						ticked();
				  }

				  //d3.select("mainEvent_" + window.current_graph_id)
				  d3.select(canvas)
					  .on("mousemove", mousemoved)
					  .call(d3.drag()
						  .container(canvas)
						  .subject(dragsubject)
						  .on("start", dragstarted)
						  .on("drag", dragged)
						  .on("end", dragended))
					  .call(zoom.on("zoom", zoomed));

				  function zoomed() {
					//console.log("zooming!");
					transform = d3.event.transform;
					//console.log(simulation.alpha());
					if(graphLayout == "force"){
						if(simulation.alpha() < 0.01){//>
						  ticked();
						}
					}else{
						ticked();
					}
				  }

				  function mousemoved() {
					//console.log(graph.nodes);
					var m = d3.mouse(this);
					var x = (m[0] - transform.x) / transform.k;
					var y = (m[1] - transform.y) / transform.k;
					var moused_node;
					for (var i = graph.nodes.length - 1; i >= 0; --i) {
					  var point = graph.nodes[i];
					  var dx = x - point.x;
					  var dy = y - point.y;
					  if (dx * dx + dy * dy < searchRadius * searchRadius) {//>
						//point.x = transform.applyX(point.x);
						//point.y = transform.applyY(point.y);
						moused_node = point;
					  }
					}
					selected_node = moused_node;
					if(graphLayout == "force"){
						if(simulation.alpha() < 0.01){//>
						  ticked();
						}
					}
				  }

				  function restartIt(){
					simulation.alpha(1.0)
					simulation.restart();
				  }

				  function ticked() {
					k = transform.k;
					if( graphLayout != "force"){
						simulation.stop();
					}else{
						if(ticks == 0){
							setTimeout(restartIt,1);
						}
					}
						context.clearRect(0, 0, width, height);
						context.save();
						//context.translate(width / 2, height / 2);

						context.globalAlpha = 1.0;

						context.translate(transform.x, transform.y);
						context.scale(transform.k, transform.k);

						graph.links.forEach(function(link) {
						  context.beginPath();
						  drawLink(link);
						  //context.strokeStyle = rgb2a("rgb(20,20,20)", get_current_alpha());
						  //console.log(link.source.color);
						  var color = mixcolors(link.source.color, link.target.color);
						  context.strokeStyle = color;
						  context.lineWidth = get_current_link_width();
						  context.stroke();
						});

						graph.nodes.forEach(function(user) {
						//console.log(user);
						  context.beginPath();
						  drawNode(user);
						  context.fillStyle = rgb2a(user.color, get_current_alpha());
						  context.fill();
						  if(selected_node != undefined){
							if(user.id == selected_node.id){
							  context.strokeStyle = "#000";
							  context.lineWidth = 1;
							  context.stroke();
							}
						  }
						  //drawLabel(user, user.id);

						});

					   graph.nodes.forEach(function(user) {
						  //if(user.deg > get_current_deg_thresh()){
							context.beginPath();
							drawLabel(user, user.id);
							  context.font="bold " + get_current_font_size() + "px Arial";
							  context.fillStyle = label_color;//user.color;
							  context.fill();
						  //}
						});

						ticks++;
						context.restore();

				  }


			}

			var payload = {};
			payload['model']    = model_name;
			payload['type']     = model_type;
			payload['memory']   = 'large';
			payload['function'] = "rankNodes";
			payload['extra_params'] = {};
			payload['extra_params']['statistic'] = colorAxis;
			//console.log(payload);

			$.ajax({
				url: url,
				type: "POST",
				data: JSON.stringify(payload), 
				beforeSend: function(xhr){
					xhr.setRequestHeader('x-api-key', API_KEY);
					xhr.setRequestHeader('Content-Type', 'application/json');
				},
				success: function(data){
					console.log(data);
					if(data['error'] == undefined){

						d3.select("#loading_status").html("loaded colorAxis data...");

						var keys = Object.keys(data['rankNodes']);

						var max_rank = 0;
						var min_rank = 0;

						graph.nodes.forEach(function(node, k){
							if(data['rankNodes'][node.id] > max_rank){
								max_rank = data['rankNodes'][node.id];
							}
							if(data['rankNodes'][node.id] < min_rank){
								min_rank = data['rankNodes'][node.id];
							}
							graph.nodes[k][colorAxis] = data['rankNodes'][node.id];
						});

						var mid_rank = (max_rank - min_rank) / 2.0;

						console.log([min_rank, mid_rank, max_rank, colormaps[colorMap][0], colormaps[colorMap][1], colormaps[colorMap][2]]);

						curr_colormap = colormap(min_rank, mid_rank, max_rank, colormaps[colorMap][0], colormaps[colorMap][1], colormaps[colorMap][2]);

						graph.nodes.forEach(function(node, k){
							graph.nodes[k].color = curr_colormap(data['rankNodes'][node.id]);
						});

						if(graphLayout == "force"){
							d3.select("#spinner_" + window.current_graph_id).style("visibility", "hidden");
							drawGraph();
							window.current_graph_id += 1
						}else{
							// bind the node embedding data to the graph!
							var max_xy = 0.0;
							var min_xy = 0.0;
							var cmax_xy, cmin_xy;
							// normalize embedding
							//console.log("ooooooooooooooooooooooooo");
							window.graph_data[window.current_graph_id]['embedding'].forEach(function(row, j){
								cmax_xy = getMaxOfArray(row);
								cmin_xy = getMinOfArray(row);
								if(cmax_xy > max_xy){
									max_xy = cmax_xy;
								}
								if(cmin_xy < min_xy){
									min_xy = cmin_xy;
								}
							});
							//console.log("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
							var range_xy = max_xy - min_xy;

							window.graph_data[window.current_graph_id]['embedding'].forEach(function(row, j){
								//console.log(row);
								graph.nodes[j]['x'] = ( ( ( Number(row[0]) ) / range_xy) * (width) )  + (width / 2);
								graph.nodes[j]['y'] = ( ( ( Number(row[1]) ) / range_xy) * (height) ) + (height / 2);
								graph.nodes[j]['fixed'] = true;
							});

							d3.select("#spinner_" + window.current_graph_id).style("visibility", "hidden");
							drawGraph();
							window.current_graph_id += 1
						}
					}
				}
			});
		});
		
		"""
		self.graph_count += 1
		return Javascript(config + javascript + js_code)
		
	
	def init_data(self):
		# check if window.graph_data already exists
		# if so then return True
		# else create window.graph_data = {} then return false
		js_code = """
			window.graph_data = [];
			window.current_graph_id = 0;
		"""
		return Javascript(js_code)
		

	def init_style(self):
		return HTML("""
		<style>
		.content_cell{
			background-color: transparent;
			background-image:       linear-gradient(0deg, transparent 24%, rgba(0, 0, 0, .05) 25%, rgba(0, 0, 0, .05) 26%, transparent 27%, transparent 74%, rgba(0, 0, 0, .05) 75%, rgba(0, 0, 0, .05) 76%, transparent 77%, transparent), linear-gradient(90deg, transparent 24%, rgba(0, 0, 0, .05) 25%, rgba(0, 0, 0, .05) 26%, transparent 27%, transparent 74%, rgba(0, 0, 0, .05) 75%, rgba(0, 0, 0, .05) 76%, transparent 77%, transparent);
		  height:100%;
		  background-size:60px 60px;
		  z-index: -1;
		}
		#header{
		  //position: absolute;
		  //top: 0;
		  //left: 0;
		  margin: 0px;
		  //height:100px;
		  width: 100%;
		  background: #fff;
		  padding:8px;
		}


		#submit_button{
		  width:50px;
		  height:50px;
		  font-size:32px;
		}

		.ui-autocomplete {
			max-height: 400px;
			overflow-y: auto;
			/* prevent horizontal scrollbar */
			overflow-x: hidden;
		  }


		#top_bar{
			position: absolute;
			//padding: 25px;
			right: 0;
			width: auto;//200px;
			height: 20px;
			z-index: 100;
			padding-left: 15px;
			padding-right:10px;
			color: rgba(255, 255, 255, 0.88);//rgba(20, 120, 220, 0.6);

			font-size: 1.0em;
			text-shadow: 5px 5px 2px rgba(120,120,120,0.2);
			background-color: rgba(20,20,20, 0.4);
			box-shadow: 5px 5px 5px rgba(0,0,0,0.2);
			border-radius: 0px 0px 0px 25px;
		}

		.loadingDialog {
		  position: absolute;
		  padding: 10px;
		  width: 350px;
		  height: 200px;
		  z-index: 15;
		  top: 50%;
		  left: 50%;
		  margin: -100px 0 0 -150px;
		  background: rgba(40,40,40,0.7);
		  border-radius: 25px 25px 25px 25px;
		  box-shadow: 5px 5px 5px rgba(0,0,0,0.2);
		}

		.loader {
		width: 50%;
			margin: 0 auto; 
			border: 16px solid #f3f3f3; /* Light grey */
			border-top: 16px solid rgb(20,120,220);
			border-right: 16px solid rgb(20,220,120);
			border-bottom: 16px solid rgb(220,20,20);
			border-radius: 50%;
			width: 120px;
			height: 120px;
			animation: spin 2s linear infinite;
		}

		#loading_status {
		color: "#fff";
		width: 50%;
			margin: 0 auto; 
		}

		@keyframes spin {
			0% { transform: rotate(0deg); }
			100% { transform: rotate(360deg); }
		}

		</style>
		""")