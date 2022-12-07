$("li.tag").on('mouseover',function(){
    $("li.tag").removeClass('active');//remove from other elements
    $(this).addClass('active');
});