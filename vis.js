require.config({
    paths: {
    d3: 'https://d3js.org/d3.v6.min'
}});

function genWordToken(d3, thisdiv, word, prob){
    var bgcolor = d3.interpolateBlues(prob)
    var textcolor = prob > 0.5 ? '#fff' : '#000'
     var div = thisdiv.append('div')
        .style('float', 'left')
         .style('margin', '2px 1px')
         .style('padding', '0px 1px 0px 1px')
         .style('display', 'inline-flex')
         .style('background-color', bgcolor)
         .style('min-width', '30px')

    var svg = div.append('svg')
         .style('width', 30)
         .style('height', 30)
         .style('margin-left', '1px')
         .style('float', 'left')
     var g = svg.append('g')

    g.append('rect')
        .attr('y', 1+20*(1-prob))
      .attr('fill', textcolor)
      .attr('width', 4)
      .attr('height', 20*prob)
      .attr('stroke-width',0 )
      .attr('stroke', '#333' )
      .attr('alignment-baseline', 'top' )
      .attr('style', 'pointer-events: none')
    g.append('text')
        .attr('x', 0)
        .attr('y', 29)
        .attr('fill', textcolor)
        .attr('font-family', 'sans-serif')
        .attr('font-size', '10px')
        .attr('text-anchor', 'left')
        .style('alignment-baseline', 'top')
        .text((prob*100).toFixed(0) + '%')

    div.append('span')
        .style('margin-left', '-23px')
        .style('margin-top', '3px')
        .style('display', 'inline-block')
        .style('pointer-events', 'none')
        .style('color', textcolor)
        .text(word)
}

function genSentToken(d3, thisdiv, prob){
    var bgcolor = d3.interpolateOrRd(prob)
    var textcolor = prob > 0.5 ? '#fff' : '#000'
     var div = thisdiv.append('div')
        .style('float', 'left')
         .style('margin', '2px 1px')
         .style('padding', '0px 1px 0px 1px')
         .style('display', 'inline-flex')
         .style('background-color', bgcolor)
         .style('min-width', '40px')

    var svg = div.append('svg')
         .style('width', 30)
         .style('height', 30)
         .style('margin-left', '1px')
         .style('float', 'left')
     var g = svg.append('g')

    g.append('rect')
        .attr('y', 1+20*(1-prob))
      .attr('fill', textcolor)
      .attr('width', 4)
      .attr('height', 20*prob)
      .attr('stroke-width',0 )
      .attr('stroke', '#333' )
      .attr('alignment-baseline', 'top' )
      .attr('style', 'pointer-events: none')
    g.append('text')
        .attr('x', 0)
        .attr('y', 29)
        .attr('fill', textcolor)
        .attr('font-family', 'sans-serif')
        .attr('font-size', '10px')
        .attr('text-anchor', 'left')
        .style('alignment-baseline', 'top')
        .text((prob*100).toFixed(0) + '%')
        
    div.append('span')
        .style('margin-left', '0px')
        .style('display', 'inline-block')
        .style('pointer-events', 'none')
        .style('color', '#ff9a00')
        .style('font-weight', 'bold')
        .text(">>")
}

function genDocToken(d3, thisdiv, word, prob){
    var bgcolor = d3.interpolateGreys(prob)
    var textcolor = prob > 0.5 ? '#fff' : '#000'
     var div = thisdiv.append('div')
        .style('float', 'left')
         .style('margin', '2px 1px')
         .style('padding', '0px 4px 0px 1px')
         .style('display', 'inline-flex')
         .style('background-color', bgcolor)
         .style('min-width', '30px')

    var svg = div.append('svg')
         .style('width', 30)
         .style('height', 30)
         .style('margin-left', '1px')
         .style('float', 'left')
     var g = svg.append('g')

    g.append('rect')
        .attr('y', 1+
        20*(1-prob))
      .attr('fill', textcolor)
      .attr('width', 4)
      .attr('height', 20*prob)
      .attr('stroke-width',0 )
      .attr('stroke', '#333' )
      .attr('alignment-baseline', 'top' )
      .attr('style', 'pointer-events: none')
    g.append('text')
        .attr('x', 0)
        .attr('y', 29)
        .attr('fill', textcolor)
        .attr('font-family', 'sans-serif')
        .attr('font-size', '10px')
        .attr('text-anchor', 'left')
        .style('alignment-baseline', 'top')
        .text((prob*100).toFixed(0) + '%')

    div.append('span')
        .style('margin-left', '-23px')
        .style('margin-top', '3px')
        .style('display', 'inline-block')
        .style('pointer-events', 'none')
        .style('color', textcolor)
        .style('font-weight', 'bold')
        .text(word)
}

function outputHAN(element, doc) {
     require(['d3'], function(d3) {
         // add figure
         var figure = d3.select(element.get(0)).append('figure')
         
        // add topline classification
         var fc = figure.append('figcaption').style('display', 'inline-block')
         genDocToken(d3, fc, doc['classification'], doc['score'])
         figure.append('br')
            
         // add each line

         for (const line of doc['lines']){
             let thisdiv = figure.append('div').style('display', 'inline-block')
             
             // sentence indicator
             genSentToken(d3, thisdiv, line['score'])
             
             // word tokens
             for (const token of line['tokens']){
                //genWordTokenBasic(d3, thisdiv, token['text'], token['score'])
                   genWordToken(d3, thisdiv, token['text'], token['score'])  
            }
            figure.append('br')
         }
     })
 };