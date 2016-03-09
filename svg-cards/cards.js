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

// green empty oval
var ovalWidth = 250;
var ovalHeight = 100;
var ovalRoundedness = ovalHeight / 2;
var oval = paper.rect(300, 250, ovalWidth, ovalHeight, ovalRoundedness);
oval.attr('fill', '#fff');
oval.attr('stroke', '#35bd2d');
oval.attr('stroke-width', '6');

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
