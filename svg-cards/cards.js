var width = 800;
var height = 600;
var paper = Raphael('container', width, height);

var background = paper.rect(0, 0, width, height);
background.attr('fill-opacity', 0.0);

var cardWidth = 350;
var cardHeight = 550;
var cardRoundedness = 20;
var cardBackground = paper.rect(250, 25, cardWidth, cardHeight, cardRoundedness);
cardBackground.attr('fill', '#fff');
cardBackground.attr('stroke', '#000');

/*
// green empty oval
var ovalWidth = 250;
var ovalHeight = 100;
var ovalRoundedness = ovalHeight / 2;
var oval = paper.rect(300, 250, ovalWidth, ovalHeight, ovalRoundedness);
oval.attr('fill', '#fff');
oval.attr('stroke', '#35bd2d');
oval.attr('stroke-width', '6');
*/

// red solid bean
var topHalfBean = paper.path(
  'M 290 300 ' +
  'c 0 0, 10 -60, 50 -60 ' +
  'c 40 0, 60 20, 100 20 ' +
  'c 70 0, 140 -70, 110 40 ' +
  'z'
);
var scaleFactor = 0.8;
topHalfBean.scale(scaleFactor, scaleFactor);
var bottomHalfBean = topHalfBean.clone();
bottomHalfBean.rotate(180);
bottomHalfBean.translate(7, -60);
var beanAttributes = {
  type: 'path',
  stroke: 'red',
  'stroke-width': '3',
  fill: 'red',
};
topHalfBean.attr(beanAttributes);
bottomHalfBean.attr(beanAttributes);

// save on "s"
$(function() {
  $('html').keypress(function(event) {
    console.log(event.which);
    if (event.which === 115) {
      var svgString = document.getElementById('container').innerHTML;
      a = document.createElement('a');
      a.download = 'card.svg';
      a.type = 'image/svg+xml';
      blob = new Blob([svgString], {"type": "image/svg+xml"});
      a.href = (window.URL || webkitURL).createObjectURL(blob);
      a.click();
    }
  });
});
